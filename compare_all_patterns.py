import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from data_manager import DataManager
from similarity import SimilarityCalculator
from recommender import CollaborativeFiltering
from evaluator import Evaluator

def evaluate_on_dataset(dataset_name, filepath, methods=['user_based', 'item_based']):
    print(f"\nEvaluating on: {dataset_name}")
    print("-" * 40)
    
    dm = DataManager()
    dm.load_from_csv(filepath)
    
    train_data, test_data = dm.create_train_test_split(test_ratio=0.2, random_seed=42)
    
    calc = SimilarityCalculator()
    recommender = CollaborativeFiltering(train_data, calc)
    evaluator = Evaluator()
    
    results = []
    
    for method in methods:
        for similarity in ['cosine', 'pearson']:
            for k in [5, 10, 20]:
                try:
                    metrics = evaluator.evaluate(
                        train_data, test_data,
                        method=method,
                        k_neighbors=k,
                        similarity_method=similarity
                    )
                    
                    results.append({
                        'dataset': dataset_name,
                        'method': method,
                        'similarity': similarity,
                        'k': k,
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'test_size': metrics['test_size']
                    })
                    
                except Exception as e:
                    print(f"  Error with {method}-{similarity}-k{k}: {e}")
    
    return results

def create_clear_comparison_visualization(all_results):
    try:
        results_df = pd.DataFrame(all_results)
        
        dataset_names = results_df['dataset'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        print("\nCreating comparison visualizations...")
        
        for idx, dataset in enumerate(dataset_names[:4]):  
            ax = axes[idx // 2, idx % 2]
            
            subset = results_df[
                (results_df['dataset'] == dataset) & 
                (results_df['similarity'] == 'cosine') &
                (results_df['k'] == 10)
            ]
            
            if not subset.empty:
                methods = subset['method'].unique()
                mae_values = []
                rmse_values = []
                labels = []
                
                for method in methods:
                    method_data = subset[subset['method'] == method]
                    if not method_data.empty:
                        mae_values.append(method_data['MAE'].iloc[0])
                        rmse_values.append(method_data['RMSE'].iloc[0])
                        labels.append(method.replace('_', ' ').title())
                
                x = np.arange(len(methods))
                width = 0.35
                
                ax.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
                ax.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
                
                ax.set_xlabel('Method')
                ax.set_ylabel('Error')
                ax.set_title(f'{dataset}\n(Cosine similarity, k=10)')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clear_comparison_results.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization: clear_comparison_results.png")

        
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, dataset in enumerate(dataset_names[:4]):
            ax = axes2[idx // 2, idx % 2]
            
            subset = results_df[results_df['dataset'] == dataset]
            
            for method in ['user_based', 'item_based']:
                method_data = subset[subset['method'] == method]
                cosine_data = method_data[method_data['similarity'] == 'cosine']
                
                if not cosine_data.empty:

                    cosine_data = cosine_data.sort_values('k')
                    ax.plot(cosine_data['k'], cosine_data['MAE'], 'o-', 
                           label=f'{method.replace("_", " ").title()}', 
                           linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Neighbors (k)')
            ax.set_ylabel('MAE')
            ax.set_title(f'{dataset}\nMAE vs k (Cosine similarity)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('mae_vs_k_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization: mae_vs_k_comparison.png")
        
        print("\n" + "="*60)
        print("WINNER ANALYSIS")
        print("="*60)
        
        for dataset in dataset_names:
            subset = results_df[
                (results_df['dataset'] == dataset) & 
                (results_df['similarity'] == 'cosine') &
                (results_df['k'] == 10)
            ]
            
            if len(subset) >= 2:
                user_mae = subset[subset['method'] == 'user_based']['MAE'].iloc[0]
                item_mae = subset[subset['method'] == 'item_based']['MAE'].iloc[0]
                
                if user_mae < item_mae:
                    winner = "USER-BASED"
                    diff = item_mae - user_mae
                else:
                    winner = "ITEM-BASED"
                    diff = user_mae - item_mae
                
                improvement = (diff / max(user_mae, item_mae)) * 100
                
                print(f"\n{dataset}:")
                print(f"  User-based MAE: {user_mae:.4f}")
                print(f"  Item-based MAE: {item_mae:.4f}")
                print(f"  Winner: {winner}")
                print(f"  Improvement: {improvement:.1f}%")
        

        try:
            plt.show()
        except:
            print("\n(Plot display not available - check .png files)")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("CLEAR COMPARISON OF COLLABORATIVE FILTERING METHODS")
    print("="*70)
    
    datasets_to_test = [
        ('User Clusters', 'patterned_data_user_clusters.csv'),
        ('Item Categories', 'patterned_data_item_categories.csv'),
        ('Sparse Data', 'sparse_data_1percent.csv'),
        ('Dense Data', 'dense_data_20percent.csv'),
        ('Clean Data', 'clean_data_lownoise.csv'),
        ('Noisy Data', 'noisy_data_highnoise.csv'),
        ('Extreme User-Based', 'extreme_user_based.csv'),
        ('Extreme Item-Based', 'extreme_item_based.csv'),
        ('Random Baseline', 'extreme_random.csv'),
    ]
    
    available_datasets = []
    for name, filepath in datasets_to_test:
        if os.path.exists(filepath):
            available_datasets.append((name, filepath))
        else:
            print(f"⚠ Dataset not found: {filepath}")
    
    if not available_datasets:
        print("\nNo patterned datasets found!")
        print("Please run: python create_patterned_data.py")
        return
    
    print(f"\nFound {len(available_datasets)} datasets for comparison")
    
    all_results = []
    
    for name, filepath in available_datasets:
        try:
            results = evaluate_on_dataset(name, filepath)
            all_results.extend(results)
            print(f"✓ Completed: {name}")
        except Exception as e:
            print(f"✗ Error with {name}: {e}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('patterned_datasets_comparison.csv', index=False)
        print(f"\n✓ Results saved to patterned_datasets_comparison.csv")
        
        create_clear_comparison_visualization(all_results)
        
        print("\n" + "="*70)
        print("SUMMARY TABLE (MAE, Cosine similarity, k=10)")
        print("="*70)
        
        summary_data = []
        for name, _ in available_datasets:
            subset = results_df[
                (results_df['dataset'] == name) & 
                (results_df['similarity'] == 'cosine') &
                (results_df['k'] == 10)
            ]
            
            if len(subset) >= 2:
                user_mae = subset[subset['method'] == 'user_based']['MAE'].iloc[0]
                item_mae = subset[subset['method'] == 'item_based']['MAE'].iloc[0]
                
                summary_data.append({
                    'Dataset': name,
                    'User-Based MAE': f"{user_mae:.4f}",
                    'Item-Based MAE': f"{item_mae:.4f}",
                    'Difference': f"{abs(user_mae - item_mae):.4f}",
                    'Better Method': 'User' if user_mae < item_mae else 'Item'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\n" + summary_df.to_string(index=False))
            
            summary_df['Difference_num'] = summary_df['Difference'].astype(float)
            max_diff_idx = summary_df['Difference_num'].idxmax()
            most_dramatic = summary_df.loc[max_diff_idx]
            
            print(f"\nMost dramatic difference:")
            print(f"  Dataset: {most_dramatic['Dataset']}")
            print(f"  Difference: {most_dramatic['Difference']}")
            print(f"  Better method: {most_dramatic['Better Method']}-based")
    
    else:
        print("\nNo results were generated")

if __name__ == "__main__":
    main()