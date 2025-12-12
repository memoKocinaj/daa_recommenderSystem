import pandas as pd
import numpy as np
import os

def create_user_clusters_data():
    np.random.seed(42)
    
    clusters = {
        'action_lovers': {'I1': 5, 'I2': 4, 'I3': 5, 'I4': 2, 'I5': 1},
        'comedy_fans': {'I1': 2, 'I2': 5, 'I3': 4, 'I4': 5, 'I5': 3},
        'drama_viewers': {'I1': 3, 'I2': 2, 'I3': 5, 'I4': 4, 'I5': 5},
    }
    
    data = []
    
    for cluster_name, item_preferences in clusters.items():
        for user_idx in range(10):  
            user_id = f'{cluster_name[0]}{user_idx+1}'  

            for item_id, base_rating in item_preferences.items():

                personal_bias = np.random.randn() * 0.5
                rating = max(1, min(5, int(round(base_rating + personal_bias))))
                data.append([user_id, item_id, rating])
            
            for _ in range(5):
                item_id = f'I{np.random.randint(6, 16)}'  
                rating = np.random.randint(1, 6)
                data.append([user_id, item_id, rating])
    
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
    df.to_csv('patterned_data_user_clusters.csv', index=False)
    print(f"Created user cluster data: {len(df)} ratings")
    print("Expected: User-based CF should work better here")
    return df

def create_item_categories_data():
    np.random.seed(43)
    
    categories = {
        'action': ['I1', 'I2', 'I3', 'I4', 'I5'],
        'comedy': ['I6', 'I7', 'I8', 'I9', 'I10'],
        'drama': ['I11', 'I12', 'I13', 'I14', 'I15'],
    }
    
    user_prefs = {
        'U1': {'action': 0.9, 'comedy': 0.1, 'drama': 0.0},
        'U2': {'action': 0.1, 'comedy': 0.9, 'drama': 0.0},
        'U3': {'action': 0.0, 'comedy': 0.1, 'drama': 0.9},
        'U4': {'action': 0.5, 'comedy': 0.3, 'drama': 0.2},
        'U5': {'action': 0.2, 'comedy': 0.5, 'drama': 0.3},
    }
    
    data = []
    
    for user_id, category_prefs in user_prefs.items():
        for category, items in categories.items():
            preference = category_prefs[category]
            
            num_items_to_rate = int(preference * 3) + 1  
            
            if num_items_to_rate > 0:
                items_to_rate = np.random.choice(items, num_items_to_rate, replace=False)
                for item_id in items_to_rate:
                    base_rating = 1 + preference * 4  
                    noise = np.random.randn() * 0.3
                    rating = max(1, min(5, int(round(base_rating + noise))))
                    data.append([user_id, item_id, rating])
        
        for _ in range(3):
            item_id = f'I{np.random.randint(16, 21)}'
            rating = np.random.randint(1, 6)
            data.append([user_id, item_id, rating])
    
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
    df.to_csv('patterned_data_item_categories.csv', index=False)
    print(f"\nCreated item category data: {len(df)} ratings")
    print("Expected: Item-based CF should work better here")
    return df

def create_sparse_vs_dense_data():
    
    np.random.seed(44)
    
    datasets = {}
    
    print("\nCreating sparse dataset (1% density)...")
    sparse_data = []
    users = [f'U{i+1}' for i in range(100)]
    items = [f'I{i+1}' for i in range(100)]
    
    num_ratings = int(100 * 100 * 0.01)  

    for _ in range(num_ratings):
        user_id = np.random.choice(users)
        item_id = np.random.choice(items)
        rating = np.random.randint(1, 6)
        sparse_data.append([user_id, item_id, rating])
    
    sparse_df = pd.DataFrame(sparse_data, columns=['user_id', 'item_id', 'rating'])
    sparse_df.to_csv('sparse_data_1percent.csv', index=False)
    datasets['sparse'] = sparse_df
    print(f"  Sparse: {len(sparse_df)} ratings, density: {len(sparse_df)/(100*100)*100:.1f}%")
    
    print("\nCreating dense dataset (20% density)...")
    dense_data = []
    users = [f'U{i+1}' for i in range(50)]
    items = [f'I{i+1}' for i in range(50)]
    
    for user_id in users:
        num_user_ratings = np.random.randint(15, 30)  
        user_items = np.random.choice(items, num_user_ratings, replace=False)
        
        for item_id in user_items:

            user_bias = np.random.uniform(-1, 1)
            rating = 3 + user_bias + np.random.randn() * 0.5
            rating = max(1, min(5, int(round(rating))))
            dense_data.append([user_id, item_id, rating])
    
    dense_df = pd.DataFrame(dense_data, columns=['user_id', 'item_id', 'rating'])
    dense_df.to_csv('dense_data_20percent.csv', index=False)
    datasets['dense'] = dense_df
    print(f"  Dense: {len(dense_df)} ratings, density: {len(dense_df)/(50*50)*100:.1f}%")
    
    return datasets

def create_noisy_vs_clean_data():
    np.random.seed(45)
    
    print("\nCreating datasets with different noise levels...")
    
    clean_data = []
    users = [f'U{i+1}' for i in range(30)]
    items = [f'I{i+1}' for i in range(40)]
    
    user_types = {}
    for i, user in enumerate(users):
        if i < 10:
            user_types[user] = {'bias': 1.0, 'noise': 0.1}  
        elif i < 20:
            user_types[user] = {'bias': 0.0, 'noise': 0.1}  
        else:
            user_types[user] = {'bias': -1.0, 'noise': 0.1}  
    
    for user_id, params in user_types.items():
        num_ratings = np.random.randint(10, 20)
        rated_items = np.random.choice(items, num_ratings, replace=False)
        
        for item_id in rated_items:

            item_factor = (int(item_id[1:]) % 10) / 10  
            
            rating = 3 + params['bias'] + item_factor + np.random.randn() * params['noise']
            rating = max(1, min(5, int(round(rating))))
            clean_data.append([user_id, item_id, rating])
    
    clean_df = pd.DataFrame(clean_data, columns=['user_id', 'item_id', 'rating'])
    clean_df.to_csv('clean_data_lownoise.csv', index=False)
    print(f"  Clean (low noise): {len(clean_df)} ratings")
    
    noisy_data = []
    for user_id, params in user_types.items():
        params['noise'] = 1.5  
        
        num_ratings = np.random.randint(10, 20)
        rated_items = np.random.choice(items, num_ratings, replace=False)
        
        for item_id in rated_items:
            rating = 3 + params['bias'] + np.random.randn() * params['noise']
            rating = max(1, min(5, int(round(rating))))
            noisy_data.append([user_id, item_id, rating])
    
    noisy_df = pd.DataFrame(noisy_data, columns=['user_id', 'item_id', 'rating'])
    noisy_df.to_csv('noisy_data_highnoise.csv', index=False)
    print(f"  Noisy (high noise): {len(noisy_df)} ratings")
    
    return {'clean': clean_df, 'noisy': noisy_df}

def create_extreme_datasets():
    print("\nCreating extreme datasets for clear visualization...")
    
    print("\n1. Perfect User-Based CF Dataset:")
    perfect_user_data = []
    
    group1_users = [f'G1U{i+1}' for i in range(15)]
    group2_users = [f'G2U{i+1}' for i in range(15)]
    
    
    
    for user in group1_users:
        for item in range(1, 11):
            rating = np.random.choice([4, 5])  
            perfect_user_data.append([user, f'I{item}', rating])
        for item in range(11, 21):
            rating = np.random.choice([1, 2])  
            perfect_user_data.append([user, f'I{item}', rating])
    
    for user in group2_users:
        for item in range(1, 11):
            rating = np.random.choice([1, 2]) 
            perfect_user_data.append([user, f'I{item}', rating])
        for item in range(11, 21):
            rating = np.random.choice([4, 5])
            perfect_user_data.append([user, f'I{item}', rating])
    
    perfect_user_df = pd.DataFrame(perfect_user_data, columns=['user_id', 'item_id', 'rating'])
    perfect_user_df.to_csv('extreme_user_based.csv', index=False)
    print(f"  Created: {len(perfect_user_df)} ratings")
    print("  Expected: User-based CF should excel, item-based should struggle")
    
    print("\n2. Perfect Item-Based CF Dataset:")
    perfect_item_data = []
    
    item_clusters = {
        'action': [f'I{i}' for i in range(1, 11)],
        'comedy': [f'I{i}' for i in range(11, 21)],
        'drama': [f'I{i}' for i in range(21, 31)]
    }
    
    users = [f'U{i+1}' for i in range(20)]
    
    for user in users:
        prefs = np.random.randn(3)
        
        for cluster_idx, (cluster_name, cluster_items) in enumerate(item_clusters.items()):
            pref = prefs[cluster_idx]
            items_to_rate = np.random.choice(cluster_items, size=3, replace=False)
            
            for item in items_to_rate:
                base_rating = 3 + pref
                rating = max(1, min(5, int(round(base_rating + np.random.randn() * 0.2))))
                perfect_item_data.append([user, item, rating])
    
    perfect_item_df = pd.DataFrame(perfect_item_data, columns=['user_id', 'item_id', 'rating'])
    perfect_item_df.to_csv('extreme_item_based.csv', index=False)
    print(f"  Created: {len(perfect_item_df)} ratings")
    print("  Expected: Item-based CF should excel")
    
    print("\n3. Random Data (Baseline):")
    random_data = []
    users = [f'RU{i+1}' for i in range(20)]
    items = [f'RI{i+1}' for i in range(30)]
    
    for _ in range(200):
        user = np.random.choice(users)
        item = np.random.choice(items)
        rating = np.random.randint(1, 6)
        random_data.append([user, item, rating])
    
    random_df = pd.DataFrame(random_data, columns=['user_id', 'item_id', 'rating'])
    random_df.to_csv('extreme_random.csv', index=False)
    print(f"  Created: {len(random_df)} ratings")
    print("  Expected: Both methods should perform equally (poorly)")
    
    return {
        'perfect_user': perfect_user_df,
        'perfect_item': perfect_item_df,
        'random': random_df
    }

def run_comparison():
    print("\n" + "="*70)
    print("CREATING DATASETS FOR CLEAR VISUALIZATION")
    print("="*70)
    
    print("\n1. Creating datasets with clear patterns...")
    user_cluster_df = create_user_clusters_data()
    item_category_df = create_item_categories_data()
    
    print("\n2. Creating datasets with different sparsity...")
    sparse_dense = create_sparse_vs_dense_data()
    
    print("\n3. Creating datasets with different noise levels...")
    noise_data = create_noisy_vs_clean_data()
    
    print("\n4. Creating extreme datasets...")
    extreme_data = create_extreme_datasets()
    
    print("\n" + "="*70)
    print("DATASETS CREATED SUCCESSFULLY")
    print("="*70)
    
    print("\nGenerated files:")
    print("1. patterned_data_user_clusters.csv - User clusters (good for user-based CF)")
    print("2. patterned_data_item_categories.csv - Item categories (good for item-based CF)")
    print("3. sparse_data_1percent.csv - Very sparse data")
    print("4. dense_data_20percent.csv - Dense data")
    print("5. clean_data_lownoise.csv - Clean data with patterns")
    print("6. noisy_data_highnoise.csv - Noisy random data")
    print("7. extreme_user_based.csv - Perfect for user-based CF")
    print("8. extreme_item_based.csv - Perfect for item-based CF")
    print("9. extreme_random.csv - Random baseline")
    
    print("\nTo compare, run:")
    print("python compare_all_patterns.py")
    
    return True

if __name__ == "__main__":

    run_comparison()
