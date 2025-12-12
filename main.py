import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_manager import DataManager
from similarity import SimilarityCalculator
from recommender import CollaborativeFiltering
from evaluator import Evaluator

def create_sample_dataset():
    print("Creating sample dataset...")
    
    #sample data
    users = [f'U{i+1}' for i in range(20)]
    items = [f'I{i+1}' for i in range(50)]
    
    ratings = []
    np.random.seed(42) 
    
    for user in users:
        num_ratings = np.random.randint(5, 16)
        rated_items = np.random.choice(items, num_ratings, replace=False)
        
        for item in rated_items:
            #ratings with some pattern
            user_pref = (ord(user[1]) + int(user[1:])) % 5  
            
            item_pop = (ord(item[1]) + int(item[1:])) % 5   

            base_rating = 2.5 + 0.5 * user_pref + 0.3 * item_pop + np.random.randn() * 0.5
            rating = max(1, min(5, round(base_rating)))
            ratings.append([user, item, float(rating)])
    
    # data saved into excel file
    df = pd.DataFrame(ratings, columns=['user_id', 'item_id', 'rating'])
    df.to_csv('sample_ratings.csv', index=False)
    print(f"Created sample_ratings.csv with {len(ratings)} ratings")
    print(f"  Users: {len(users)}, Items: {len(items)}")
    print(f"  Sparsity: {len(ratings)/(len(users)*len(items))*100:.1f}% of possible ratings")
    
    return df

def load_movielens_data():

    movielens_paths = [
        'ml-latest-small/ratings.csv',
        'movielens/ratings.csv',
        'data/ratings.csv'
    ]
    
    for path in movielens_paths:
        if os.path.exists(path):
            print(f"Loading MovieLens data from {path}...")
            df = pd.read_csv(path)
            # Rename columns if needed
            if 'userId' in df.columns and 'movieId' in df.columns:
                df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
            return df
    
    print("MovieLens dataset not found. Using sample data instead.")
    return None

def visualize_results(results_df):
    try:
        if results_df.empty:
            print("No results to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plotting MAE
        for method in results_df['method'].unique():
            for sim in results_df['similarity'].unique():
                mask = (results_df['method'] == method) & (results_df['similarity'] == sim)
                subset = results_df[mask]
                
                if not subset.empty:
                    label = f"{method} ({sim})"
                    axes[0].plot(subset['k'], subset['MAE'], 'o-', label=label, linewidth=2)
                    axes[1].plot(subset['k'], subset['RMSE'], 'o-', label=label, linewidth=2)
        
        axes[0].set_xlabel('Number of Neighbors (k)')
        axes[0].set_ylabel('Mean Absolute Error (MAE)')
        axes[0].set_title('MAE vs. Number of Neighbors')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Number of Neighbors (k)')
        axes[1].set_ylabel('Root Mean Square Error (RMSE)')
        axes[1].set_title('RMSE vs. Number of Neighbors')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to evaluation_results.png")
        
        #displays
        try:
            plt.show()
        except:
            print("Note: Plot display not available in this environment")
            print("      Check evaluation_results.png file for the visualization")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

def interactive_mode(recommender, train_data):
    """Interactive mode for testing recommendations"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  recommend <user_id> <method> <n> - Get top N recommendations")
    print("    method: user_based or item_based")
    print("    example: recommend U1 user_based 5")
    print("  predict <user_id> <item_id> <method> - Predict rating")
    print("    example: predict U1 I10 user_based")
    print("  users - List all users")
    print("  items - List all items")
    print("  stats - Show dataset statistics")
    print("  quit - Exit interactive mode")
    print("="*60)
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            elif command.lower() == 'users':
                users = train_data.user_ids
                print(f"Total users: {len(users)}")
                print("First 20 users:", users[:20])
                if len(users) > 20:
                    print(f"... and {len(users) - 20} more")
            
            elif command.lower() == 'items':
                items = train_data.item_ids
                print(f"Total items: {len(items)}")
                print("First 20 items:", items[:20])
                if len(items) > 20:
                    print(f"... and {len(items) - 20} more")
            
            elif command.lower() == 'stats':
                total_ratings = sum(len(v) for v in train_data.user_ratings.values())
                print(f"Dataset Statistics:")
                print(f"  Users: {len(train_data.user_ids)}")
                print(f"  Items: {len(train_data.item_ids)}")
                print(f"  Ratings: {total_ratings}")
                print(f"  Sparsity: {total_ratings/(len(train_data.user_ids)*len(train_data.item_ids))*100:.2f}%")
                
                avg_ratings_per_user = total_ratings / len(train_data.user_ids)
                print(f"  Avg ratings per user: {avg_ratings_per_user:.1f}")
            
            elif command.lower().startswith('recommend'):
                parts = command.split()
                if len(parts) >= 3:
                    user_id = parts[1]
                    method = parts[2] if len(parts) > 2 else 'user_based'
                    n = int(parts[3]) if len(parts) > 3 else 5
                    
                    if user_id not in train_data.user_ids:
                        print(f"✗ User '{user_id}' not found in training data")
                        print(f"  Available users: {train_data.user_ids[:10]}...")
                        continue
                    
                    if method not in ['user_based', 'item_based']:
                        print(f"✗ Method must be 'user_based' or 'item_based'")
                        continue
                    
                    print(f"Getting top {n} recommendations for user {user_id} ({method})...")
                    try:
                        recommendations = recommender.get_top_n_recommendations(
                            user_id, n=n, method=method,
                            k_neighbors=10, similarity_method='cosine'
                        )
                        
                        if not recommendations:
                            print(f"  No recommendations available")
                        else:
                            print(f"\nTop {n} recommendations:")
                            for i, (item_id, pred_rating) in enumerate(recommendations, 1):
                                print(f"  {i:2}. Item {item_id:5} - Predicted rating: {pred_rating:.2f}")
                    
                    except Exception as e:
                        print(f"✗ Error getting recommendations: {e}")
                else:
                    print("Usage: recommend <user_id> <method> <n>")
                    print("Example: recommend U1 user_based 5")
            
            elif command.lower().startswith('predict'):
                parts = command.split()
                if len(parts) >= 4:
                    user_id = parts[1]
                    item_id = parts[2]
                    method = parts[3]
                    
                    if user_id not in train_data.user_ids:
                        print(f"✗ User '{user_id}' not found")
                        continue
                    
                    if method not in ['user_based', 'item_based']:
                        print(f"✗ Method must be 'user_based' or 'item_based'")
                        continue
                    
                    print(f"Predicting rating for user {user_id} on item {item_id} ({method})...")
                    try:
                        if method == 'user_based':
                            pred = recommender.predict_rating_user_based(
                                user_id, item_id, k=10, method='cosine'
                            )
                        else:
                            pred = recommender.predict_rating_item_based(
                                user_id, item_id, k=10, method='cosine'
                            )
                        
                        # Check if user already rated this item
                        if user_id in train_data.user_ratings and item_id in train_data.user_ratings[user_id]:
                            actual = train_data.user_ratings[user_id][item_id]
                            print(f"  User already rated this item: {actual:.2f}")
                            print(f"  Predicted rating: {pred:.2f}")
                            print(f"  Difference: {abs(pred - actual):.2f}")
                        else:
                            print(f"  Predicted rating: {pred:.2f}")
                    
                    except Exception as e:
                        print(f"✗ Error predicting rating: {e}")
                else:
                    print("Usage: predict <user_id> <item_id> <method>")
                    print("Example: predict U1 I10 user_based")
            
            else:
                print("Unknown command. Available commands:")
                print("  recommend, predict, users, items, stats, quit")
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_demo(recommender, train_data, test_data):
    """Run a demonstration of the system"""
    print("\n" + "="*60)
    print("SYSTEM DEMONSTRATION")
    print("="*60)
    
    sample_user = train_data.user_ids[0] if train_data.user_ids else None
    if not sample_user:
        print("No users in dataset")
        return
    
    print(f"\nSample user: {sample_user}")
    
    user_ratings = train_data.user_ratings.get(sample_user, {})
    if user_ratings:
        print(f"\nUser {sample_user}'s ratings ({len(user_ratings)} items):")
        for item_id, rating in list(user_ratings.items())[:10]:  # Show first 10
            print(f"  Item {item_id}: {rating}")
        if len(user_ratings) > 10:
            print(f"  ... and {len(user_ratings) - 10} more")
    else:
        print(f"User {sample_user} has no ratings")
    
    print("\n" + "-"*40)
    print("Generating recommendations...")
    
    methods = ['user_based', 'item_based']
    for method in methods:
        print(f"\n{method.replace('_', ' ').title()} recommendations:")
        try:
            recs = recommender.get_top_n_recommendations(
                sample_user, n=5, method=method,
                k_neighbors=10, similarity_method='cosine'
            )
            
            if recs:
                for i, (item_id, pred_rating) in enumerate(recs, 1):
                    print(f"  {i}. Item {item_id}: predicted rating = {pred_rating:.2f}")
            else:
                print("  No recommendations available")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "-"*40)
    print("Comparing prediction methods:")
    
#finds then rates what is left out
    user_rated_items = set(user_ratings.keys())
    all_items = set(train_data.item_ids)
    unrated_items = list(all_items - user_rated_items)
    
    if unrated_items:
        test_item = unrated_items[0]
        print(f"\nPredicting rating for item {test_item}:")
        
        user_pred = recommender.predict_rating_user_based(
            sample_user, test_item, k=10, method='cosine'
        )
        item_pred = recommender.predict_rating_item_based(
            sample_user, test_item, k=10, method='cosine'
        )
        
        print(f"  User-based prediction: {user_pred:.2f}")
        print(f"  Item-based prediction: {item_pred:.2f}")
        print(f"  Difference: {abs(user_pred - item_pred):.2f}")
    else:
        print("User has rated all items!")















def main():
    print("=" * 70)
    print("COLLABORATIVE FILTERING RECOMMENDER SYSTEM")
    print("Design and Analysis of Algorithms Project")
    print("=" * 70)
    
    print("\n[1] LOADING DATA")
    print("-" * 40)
    
    data_manager = DataManager()
    
    movielens_data = load_movielens_data()
    
    if movielens_data is not None:
        movielens_data[['user_id', 'item_id', 'rating']].to_csv('current_ratings.csv', index=False)
        data_manager.load_from_csv('current_ratings.csv')
        print(f"✓ Loaded MovieLens dataset")

    else:
        create_sample_dataset()
        data_manager.load_from_csv('sample_ratings.csv')
        print(f"✓ Loaded sample dataset")
    

    print("\n[2] PREPARING DATA")
    print("-" * 40)
    
    print("Splitting data into training and test sets...")
    train_data, test_data = data_manager.create_train_test_split(test_ratio=0.2, random_seed=42)
    
    print(f"✓ Training set: {len(train_data.user_ids)} users, "
          f"{sum(len(v) for v in train_data.user_ratings.values())} ratings")
    print(f"✓ Test set: {len(test_data.user_ids)} users, "
          f"{sum(len(v) for v in test_data.user_ratings.values())} ratings")


    
    print("\n[3] INITIALIZING SYSTEM COMPONENTS")
    print("-" * 40)
    
    similarity_calc = SimilarityCalculator()
    recommender = CollaborativeFiltering(train_data, similarity_calc)
    evaluator = Evaluator()
    
    print("✓ DataManager initialized")
    print("✓ SimilarityCalculator initialized")
    print("✓ CollaborativeFiltering initialized")
    print("✓ Evaluator initialized")
    
    run_demo(recommender, train_data, test_data)
    
    print("\n[4] EVALUATING SYSTEM PERFORMANCE")
    print("-" * 40)
    
    print("Running evaluations with different configurations...")
    results = evaluator.compare_methods(
        train_data, test_data,
        k_values=[5, 10, 20],
        similarity_methods=['cosine', 'pearson']
    )
    
    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print("-" * 40)
    print(results_df[['method', 'similarity', 'k', 'MAE', 'RMSE', 'test_size']].to_string())
    
    best_idx = results_df['MAE'].idxmin()
    best_result = results_df.loc[best_idx]
    print(f"\n✓ Best configuration:")
    print(f"  Method: {best_result['method']}")
    print(f"  Similarity: {best_result['similarity']}")
    print(f"  k: {best_result['k']}")
    print(f"  MAE: {best_result['MAE']:.4f}")
    print(f"  RMSE: {best_result['RMSE']:.4f}")
    
    print("\n[5] VISUALIZING RESULTS")
    print("-" * 40)
    visualize_results(results_df)
    
    print("\n[6] STARTING INTERACTIVE MODE")
    interactive_mode(recommender, train_data)
    
    print("\n" + "=" * 70)
    print("PROGRAM COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - sample_ratings.csv: Sample dataset")
    if os.path.exists('evaluation_results.png'):
        print(f"  - evaluation_results.png: Performance visualization")
    print(f"  - Various .py files: Source code modules")
    print("\nTo run again, simply execute: python main.py")



def run_quick_test():
    print("Running quick system test...")
    
    try:
        from data_manager import DataManager
        from similarity import SimilarityCalculator
        from recommender import CollaborativeFiltering
        from evaluator import Evaluator
        
        print("✓ All imports successful")
        
        dm = DataManager()
        test_data = [
            ['U1', 'I1', 5],
            ['U1', 'I2', 3],
            ['U2', 'I1', 4],
            ['U2', 'I3', 2],
        ]
        
        import pandas as pd
        df = pd.DataFrame(test_data, columns=['user_id', 'item_id', 'rating'])
        df.to_csv('quick_test.csv', index=False)
        
        dm.load_from_csv('quick_test.csv')
        print(f"✓ Data loaded: {len(dm.user_ids)} users, {len(dm.item_ids)} items")
        
        calc = SimilarityCalculator()
        sim = calc.user_similarity(dm, 'U1', 'U2', method='cosine')
        print(f"✓ Similarity calculation: U1-U2 = {sim:.4f}")
        
        import os
        if os.path.exists('quick_test.csv'):
            os.remove('quick_test.csv')
        
        print("\n✓ Quick test passed! System is ready.")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        if run_quick_test():

            print("\n" + "="*70)
            response = input("Run full recommender system? (y/n): ").strip().lower()
            if response in ['y', 'yes', '']:
                main()
            else:
                print("Exiting. Run 'python main.py' later to start the full program.")
        else:
            print("\nSystem test failed. Please check your installation and modules.")
            
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError in main program: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check that all required modules are in the same directory.")