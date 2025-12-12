import numpy as np
from typing import Dict, List, Tuple, Optional
from data_manager import DataManager

class SimilarityCalculator:
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = np.sqrt(sum(v * v for v in vec1))
        norm2 = np.sqrt(sum(v * v for v in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def pearson_correlation(vec1: List[float], vec2: List[float]) -> float:
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        if len(vec1) < 2:
            return 0.0
        
        mean1 = np.mean(vec1)
        mean2 = np.mean(vec2)
        
        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(vec1, vec2))
        denom1 = sum((v1 - mean1) ** 2 for v1 in vec1)
        denom2 = sum((v2 - mean2) ** 2 for v2 in vec2)
        
        if denom1 == 0 or denom2 == 0:
            return 0.0
        
        return numerator / np.sqrt(denom1 * denom2)
    
    def user_similarity(self, data: DataManager, user1_id: str, user2_id: str, 
                       method: str = 'cosine') -> float:
        
        common_items = data.get_common_items(user1_id, user2_id)
        
        if len(common_items) < 2:
            return 0.0
        
        ratings1 = [data.user_ratings[user1_id][item] for item in common_items]
        ratings2 = [data.user_ratings[user2_id][item] for item in common_items]
        
        if method == 'cosine':
            return self.cosine_similarity(ratings1, ratings2)
        elif method == 'pearson':
            return self.pearson_correlation(ratings1, ratings2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def item_similarity(self, data: DataManager, item1_id: str, item2_id: str,
                       method: str = 'cosine') -> float:
       
        common_users = data.get_common_users(item1_id, item2_id)
        
        if len(common_users) < 2:
            return 0.0
        
        ratings1 = [data.item_ratings[item1_id][user] for user in common_users]
        ratings2 = [data.item_ratings[item2_id][user] for user in common_users]
        
        if method == 'cosine':
            return self.cosine_similarity(ratings1, ratings2)
        elif method == 'pearson':
            return self.pearson_correlation(ratings1, ratings2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def compute_user_similarity_matrix(self, data: DataManager, 
                                      method: str = 'cosine',
                                      max_users: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        
        user_ids = data.user_ids
        if max_users and max_users < len(user_ids):
            user_ids = user_ids[:max_users]
        
        print(f"Computing user similarity matrix for {len(user_ids)} users...")
        similarity_matrix = {uid: {} for uid in user_ids}
        
        for i, user1 in enumerate(user_ids):
            similarity_matrix[user1][user1] = 1.0  
            
            for j in range(i + 1, len(user_ids)):
                user2 = user_ids[j]
                sim = self.user_similarity(data, user1, user2, method)
                similarity_matrix[user1][user2] = sim
                similarity_matrix[user2][user1] = sim
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(user_ids)} users")
        
        print("Similarity matrix computation complete")
        return similarity_matrix
    
    def compute_item_similarity_matrix(self, data: DataManager,
                                      method: str = 'cosine',
                                      max_items: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        
        item_ids = data.item_ids
        if max_items and max_items < len(item_ids):
            item_ids = item_ids[:max_items]
        
        print(f"Computing item similarity matrix for {len(item_ids)} items...")
        similarity_matrix = {iid: {} for iid in item_ids}
        
        for i, item1 in enumerate(item_ids):
            similarity_matrix[item1][item1] = 1.0  
            
            for j in range(i + 1, len(item_ids)):
                item2 = item_ids[j]
                sim = self.item_similarity(data, item1, item2, method)
                similarity_matrix[item1][item2] = sim
                similarity_matrix[item2][item1] = sim
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(item_ids)} items")
        
        print("Item similarity matrix computation complete")
        return similarity_matrix