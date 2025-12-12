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
    
    users = [f'U{i+1}' for i in range(20)]
    items = [f'I{i+1}' for i in range(50)]
    ratings = []
    np.random.seed(42)
    
    for user in users:
        num_ratings = np.random.randint(5, 16)
        rated_items = np.random.choice(items, num_ratings, replace=False)
        for item in rated_items:
            user_pref = (ord(user[1]) + int(user[1:])) % 5
            item_pop = (ord(item[1]) + int(item[1:])) % 5
            rating = 2.5 + 0.5*user_pref + 0.3*item_pop + np.random.randn()*0.5
            rating = max(1, min(5, round(rating)))
            ratings.append([user, item, float(rating)])
    
    df = pd.DataFrame(ratings, columns=['user_id', 'item_id', 'rating'])
    df.to_csv('sample_ratings.csv', index=False)
    total_possible = len(users) * len(items)
    sparsity = len(ratings)/total_possible*100
    print(f"✓ Created sample_ratings.csv: {len(ratings)} ratings")
    print(f"  Users: {len(users)}, Items: {len(items)}, Sparsity: {sparsity:.1f}%")
    return df

def load_movielens_data():
    paths = ['ml-latest-small/ratings.csv', 'movielens/ratings.csv', 'data/ratings.csv']
    for path in paths:
        if os.path.exists(path):
            print(f"Loading MovieLens data from {path}...")
            df = pd.read_csv(path)
            if 'userId' in df.columns and 'movieId' in df.columns:
                df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
            return df
    print("MovieLens dataset not found. Using sample data.")
    return None

def visualize_results(results_df):
    if results_df.empty:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for method in results_df['method'].unique():
        for sim in results_df['similarity'].unique():
            subset = results_df[(results_df['method'] == method) & (results_df['similarity'] == sim)]
            if not subset.empty:
                label = f"{method} ({sim})"
                axes[0].plot(subset['k'], subset['MAE'], 'o-', label=label, linewidth=2)
                axes[1].plot(subset['k'], subset['RMSE'], 'o-', label=label, linewidth=2)
    
    for ax, metric, title in [(axes[0], 'MAE', 'Mean Absolute Error'), 
                               (axes[1], 'RMSE', 'Root Mean Square Error')]:
        ax.set_xlabel('Number of Neighbors (k)')
        ax.set_ylabel(metric)
        ax.set_title(f'{title} vs. Number of Neighbors')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to evaluation_results.png")
    
    try:
        plt.show()
    except:
        print("Note: Plot display not available - check evaluation_results.png")

def interactive_mode(recommender, train_data):
    commands = """
Commands:
  recommend <user_id> <method> <n> - Get top N recommendations
    method: user_based or item_based
    example: recommend U1 user_based 5
  predict <user_id> <item_id> <method> - Predict rating
    example: predict U1 I10 user_based
  users - List all users
  items - List all items
  stats - Show dataset statistics
  quit - Exit interactive mode
"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60 + commands + "="*60)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            elif cmd == 'users':
                users = train_data.user_ids
                print(f"Total users: {len(users)}")
                print("First 20:", users[:20])
                if len(users) > 20:
                    print(f"... and {len(users) - 20} more")
            
            elif cmd == 'items':
                items = train_data.item_ids
                print(f"Total items: {len(items)}")
                print("First 20:", items[:20])
                if len(items) > 20:
                    print(f"... and {len(items) - 20} more")
            
            elif cmd == 'stats':
                total_ratings = sum(len(v) for v in train_data.user_ratings.values())
                total_possible = len(train_data.user_ids) * len(train_data.item_ids)
                sparsity = total_ratings/total_possible*100
                stats = {
                    'Users': len(train_data.user_ids),
                    'Items': len(train_data.item_ids),
                    'Ratings': total_ratings,
                    'Sparsity': f"{sparsity:.2f}%",
                    'Avg ratings/user': f"{total_ratings/len(train_data.user_ids):.1f}"
                }
                print("Dataset Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            
            elif cmd.startswith('recommend'):
                parts = cmd.split()
                if len(parts) >= 3:
                    user_id, method = parts[1], parts[2]
                    n = int(parts[3]) if len(parts) > 3 else 5
                    
                    if user_id not in train_data.user_ids:
                        print(f"✗ User '{user_id}' not found")
                        print(f"  Available: {train_data.user_ids[:10]}...")
                        continue
                    
                    if method not in ['user_based', 'item_based']:
                        print("✗ Method must be 'user_based' or 'item_based'")
                        continue
                    
                    print(f"Getting top {n} recommendations for {user_id} ({method})...")
                    try:
                        recs = recommender.get_top_n_recommendations(
                            user_id, n=n, method=method, k_neighbors=10, 
                            similarity_method='cosine'
                        )
                        
                        if recs:
                            print(f"\nTop {n} recommendations:")
                            for i, (item_id, pred_rating) in enumerate(recs, 1):
                                print(f"  {i:2}. Item {item_id:5} - Predicted: {pred_rating:.2f}")
                        else:
                            print("No recommendations available")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                else:
                    print("Usage: recommend <user_id> <method> <n>")
            
            elif cmd.startswith('predict'):
                parts = cmd.split()
                if len(parts) >= 4:
                    user_id, item_id, method = parts[1], parts[2], parts[3]
                    
                    if user_id not in train_data.user_ids:
                        print(f"✗ User '{user_id}' not found")
                        continue
                    
                    if method not in ['user_based', 'item_based']:
                        print("✗ Method must be 'user_based' or 'item_based'")
                        continue
                    
                    print(f"Predicting rating for {user_id} on {item_id} ({method})...")
                    try:
                        if method == 'user_based':
                            pred = recommender.predict_rating_user_based(
                                user_id, item_id, k=10, method='cosine'
                            )
                        else:
                            pred = recommender.predict_rating_item_based(
                                user_id, item_id, k=10, method='cosine'
                            )
                        
                        if user_id in train_data.user_ratings and item_id in train_data.user_ratings[user_id]:
                            actual = train_data.user_ratings[user_id][item_id]
                            print(f"  Actual: {actual:.2f}, Predicted: {pred:.2f}")
                            print(f"  Difference: {abs(pred - actual):.2f}")
                        else:
                            print(f"  Predicted rating: {pred:.2f}")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                else:
                    print("Usage: predict <user_id> <item_id> <method>")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_demo(recommender, train_data, test_data):
    print("\n" + "="*60)
    print("SYSTEM DEMONSTRATION")
    print("="*60)
    
    if not train_data.user_ids:
        print("No users in dataset")
        return
    
    sample_user = train_data.user_ids[0]
    user_ratings = train_data.user_ratings.get(sample_user, {})
    
    print(f"\nSample user: {sample_user}")
    if user_ratings:
        print(f"\nUser's ratings ({len(user_ratings)} items):")
        for item_id, rating in list(user_ratings.items())[:10]:
            print(f"  Item {item_id}: {rating}")
        if len(user_ratings) > 10:
            print(f"  ... and {len(user_ratings) - 10} more")
    else:
        print(f"User {sample_user} has no ratings")
    
    print("\n" + "-"*40)
    print("Generating recommendations...")
    
    for method in ['user_based', 'item_based']:
        print(f"\n{method.replace('_', ' ').title()} recommendations:")
        try:
            recs = recommender.get_top_n_recommendations(
                sample_user, n=5, method=method,
                k_neighbors=10, similarity_method='cosine'
            )
            for i, (item_id, pred_rating) in enumerate(recs, 1):
                print(f"  {i}. Item {item_id}: {pred_rating:.2f}")
        except Exception as e:
            print(f"  Error: {e}")

def run_quick_test():
    print("Running quick system test...")
    try:
        dm = DataManager()
        test_data = [
            ['U1', 'I1', 5], ['U1', 'I2', 3],
            ['U2', 'I1', 4], ['U2', 'I3', 2],
        ]
        df = pd.DataFrame(test_data, columns=['user_id', 'item_id', 'rating'])
        df.to_csv('quick_test.csv', index=False)
        
        dm.load_from_csv('quick_test.csv')
        calc = SimilarityCalculator()
        sim = calc.user_similarity(dm, 'U1', 'U2', method='cosine')
        
        if os.path.exists('quick_test.csv'):
            os.remove('quick_test.csv')
        
        print(f"✓ Quick test passed! Similarity U1-U2 = {sim:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False

def main():
    print("=" * 70)
    print("COLLABORATIVE FILTERING RECOMMENDER SYSTEM")
    print("Design and Analysis of Algorithms Project")
    print("=" * 70)
    
    print("\n[1] LOADING DATA")
    print("-" * 40)
    
    dm = DataManager()
    movielens_data = load_movielens_data()
    
    if movielens_data is not None:
        movielens_data[['user_id', 'item_id', 'rating']].to_csv('current_ratings.csv', index=False)
        dm.load_from_csv('current_ratings.csv')
        print("✓ Loaded MovieLens dataset")
    else:
        create_sample_dataset()
        dm.load_from_csv('sample_ratings.csv')
        print("✓ Loaded sample dataset")
    
    print("\n[2] PREPARING DATA")
    print("-" * 40)
    
    train_data, test_data = dm.create_train_test_split(test_ratio=0.2, random_seed=42)
    train_ratings = sum(len(v) for v in train_data.user_ratings.values())
    test_ratings = sum(len(v) for v in test_data.user_ratings.values())
    print(f"✓ Training: {len(train_data.user_ids)} users, {train_ratings} ratings")
    print(f"✓ Test: {len(test_data.user_ids)} users, {test_ratings} ratings")
    
    print("\n[3] INITIALIZING SYSTEM")
    print("-" * 40)
    
    similarity_calc = SimilarityCalculator()
    recommender = CollaborativeFiltering(train_data, similarity_calc)
    evaluator = Evaluator()
    print("✓ All components initialized")
    
    run_demo(recommender, train_data, test_data)
    
    print("\n[4] EVALUATING PERFORMANCE")
    print("-" * 40)
    
    print("Running evaluations...")
    results = evaluator.compare_methods(
        train_data, test_data,
        k_values=[5, 10, 20],
        similarity_methods=['cosine', 'pearson']
    )
    
    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print("-" * 40)
    print(results_df[['method', 'similarity', 'k', 'MAE', 'RMSE']].to_string())
    
    best = results_df.loc[results_df['MAE'].idxmin()]
    print(f"\n✓ Best configuration:")
    print(f"  Method: {best['method']}, Similarity: {best['similarity']}, k: {best['k']}")
    print(f"  MAE: {best['MAE']:.4f}, RMSE: {best['RMSE']:.4f}")
    
    print("\n[5] VISUALIZING RESULTS")
    print("-" * 40)
    visualize_results(results_df)
    
    print("\n[6] INTERACTIVE MODE")
    interactive_mode(recommender, train_data)
    
    print("\n" + "=" * 70)
    print("PROGRAM COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    try:
        if run_quick_test():
            response = input("\nRun full recommender system? (y/n): ").strip().lower()
            if response in ['y', 'yes', '']:
                main()
            else:
                print("Exiting. Run 'python main.py' later.")
        else:
            print("\nSystem test failed. Check installation.")
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except Exception as e:
        print(f"\nError: {e}")