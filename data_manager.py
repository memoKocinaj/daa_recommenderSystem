import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import json

class DataManager:
    
    def __init__(self):
        self.user_ratings = {}  
        self.item_ratings = {}  
        self.user_ids = []
        self.item_ids = []
        self.user_index_map = {}
        self.item_index_map = {}
        
    def load_from_csv(self, filepath: str, user_col: str = 'user_id', 
                      item_col: str = 'item_id', rating_col: str = 'rating'):
        
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        self.user_ratings = {}
        self.item_ratings = {}
        
        for _, row in df.iterrows():
            user_id = str(row[user_col])
            item_id = str(row[item_col])
            rating = float(row[rating_col])
            
            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}
            self.user_ratings[user_id][item_id] = rating
            
            if item_id not in self.item_ratings:
                self.item_ratings[item_id] = {}
            self.item_ratings[item_id][user_id] = rating
        
        self.user_ids = list(self.user_ratings.keys())
        self.item_ids = list(self.item_ratings.keys())
        self.user_index_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_index_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        print(f"Loaded {len(self.user_ids)} users, {len(self.item_ids)} items, "
              f"{df.shape[0]} ratings")
        return self
    
    def create_train_test_split(self, test_ratio: float = 0.2, random_seed: int = 42):
        np.random.seed(random_seed)
        
        train_data = DataManager()
        test_data = DataManager()
        
        train_data.user_ratings = {}
        test_data.user_ratings = {}
        
        for user_id, item_ratings in self.user_ratings.items():
            train_items = {}
            test_items = {}
            
            items = list(item_ratings.keys())
            ratings = list(item_ratings.values())
            
            if len(items) > 1: 
                test_size = max(1, int(len(items) * test_ratio))
                test_indices = np.random.choice(len(items), test_size, replace=False)
                
                for idx, (item_id, rating) in enumerate(item_ratings.items()):
                    if idx in test_indices:
                        test_items[item_id] = rating
                    else:
                        train_items[item_id] = rating
            else:
                train_items = item_ratings.copy()
            
            if train_items:
                train_data.user_ratings[user_id] = train_items
            if test_items:
                test_data.user_ratings[user_id] = test_items
        
        train_data._build_item_ratings()
        test_data._build_item_ratings()
        
        train_data._build_mappings()
        test_data._build_mappings()
        
        print(f"Train set: {sum(len(v) for v in train_data.user_ratings.values())} ratings")
        print(f"Test set: {sum(len(v) for v in test_data.user_ratings.values())} ratings")
        
        return train_data, test_data
    
    def get_user_average_rating(self, user_id: str) -> float:
        if user_id not in self.user_ratings:
            return 0.0
        ratings = list(self.user_ratings[user_id].values())
        return sum(ratings) / len(ratings) if ratings else 0.0
    
    def get_item_average_rating(self, item_id: str) -> float:
        if item_id not in self.item_ratings:
            return 0.0
        ratings = list(self.item_ratings[item_id].values())
        return sum(ratings) / len(ratings) if ratings else 0.0
    
    def get_common_items(self, user1_id: str, user2_id: str) -> List[str]:
        items1 = set(self.user_ratings.get(user1_id, {}).keys())
        items2 = set(self.user_ratings.get(user2_id, {}).keys())
        return list(items1.intersection(items2))
    
    def get_common_users(self, item1_id: str, item2_id: str) -> List[str]:
        users1 = set(self.item_ratings.get(item1_id, {}).keys())
        users2 = set(self.item_ratings.get(item2_id, {}).keys())
        return list(users1.intersection(users2))
    
    def _build_item_ratings(self):
        self.item_ratings = {}
        for user_id, items in self.user_ratings.items():
            for item_id, rating in items.items():
                if item_id not in self.item_ratings:
                    self.item_ratings[item_id] = {}
                self.item_ratings[item_id][user_id] = rating
    
    def _build_mappings(self):
        self.user_ids = list(self.user_ratings.keys())
        self.item_ids = list(self.item_ratings.keys())
        self.user_index_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_index_map = {iid: i for i, iid in enumerate(self.item_ids)}
    
    def save_to_json(self, filepath: str):
        data = {
            'user_ratings': self.user_ratings,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_from_json(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.user_ratings = {k: {ik: float(iv) for ik, iv in v.items()} 
                            for k, v in data['user_ratings'].items()}
        self.user_ids = data['user_ids']
        self.item_ids = data['item_ids']
        self._build_item_ratings()
        self._build_mappings()
        return self