import numpy as np
from typing import Dict, List, Tuple, Optional
from data_manager import DataManager
from similarity import SimilarityCalculator

class CollaborativeFiltering:
    
    
    def __init__(self, data: DataManager, similarity_calc: SimilarityCalculator):
        self.data = data
        self.similarity_calc = similarity_calc
        self.user_sim_matrix = None
        self.item_sim_matrix = None
        
    def find_similar_users(self, user_id: str, k: int = 20, 
                          method: str = 'cosine') -> List[Tuple[str, float]]:
        similarities = []
        
        for other_user in self.data.user_ids:
            if other_user == user_id:
                continue
            
            sim = self.similarity_calc.user_similarity(
                self.data, user_id, other_user, method
            )
            
            if sim > 0:   
                similarities.append((other_user, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def find_similar_items(self, item_id: str, k: int = 20,
                          method: str = 'cosine') -> List[Tuple[str, float]]:
        similarities = []
        
        for other_item in self.data.item_ids:
            if other_item == item_id:
                continue
            
            sim = self.similarity_calc.item_similarity(
                self.data, item_id, other_item, method
            )
            
            if sim > 0:
                similarities.append((other_item, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def predict_rating_user_based(self, user_id: str, item_id: str, 
                                 k: int = 20, method: str = 'cosine') -> float:

        if user_id in self.data.user_ratings and item_id in self.data.user_ratings[user_id]:
            return self.data.user_ratings[user_id][item_id]
        
        user_avg = self.data.get_user_average_rating(user_id)
        
        users_who_rated = self.data.item_ratings.get(item_id, {})
        if not users_who_rated:
            return user_avg  
        
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for other_user, rating in users_who_rated.items():
            if other_user == user_id:
                continue
                
            sim = self.similarity_calc.user_similarity(
                self.data, user_id, other_user, method
            )
            
            if sim > 0:  
                other_user_avg = self.data.get_user_average_rating(other_user)
                weighted_sum += sim * (rating - other_user_avg)
                similarity_sum += abs(sim)
        
        if similarity_sum == 0:
            return user_avg
        
        predicted = user_avg + (weighted_sum / similarity_sum)
        return max(1.0, min(5.0, predicted))
    
    def predict_rating_item_based(self, user_id: str, item_id: str,
                                 k: int = 20, method: str = 'cosine') -> float:
 
        if user_id in self.data.user_ratings and item_id in self.data.user_ratings[user_id]:
            return self.data.user_ratings[user_id][item_id]
        
        item_avg = self.data.get_item_average_rating(item_id)
        
        items_rated_by_user = self.data.user_ratings.get(user_id, {})
        if not items_rated_by_user:
            return item_avg  
        
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for other_item, rating in items_rated_by_user.items():
            if other_item == item_id:
                continue
                
            sim = self.similarity_calc.item_similarity(
                self.data, item_id, other_item, method
            )
            
            if sim > 0:
                other_item_avg = self.data.get_item_average_rating(other_item)
                weighted_sum += sim * (rating - other_item_avg)
                similarity_sum += abs(sim)
        
        if similarity_sum == 0:
            return item_avg
        
        predicted = item_avg + (weighted_sum / similarity_sum)
        return max(1.0, min(5.0, predicted))
    
    def get_top_n_recommendations(self, user_id: str, n: int = 10,
                                 method: str = 'user_based',
                                 k_neighbors: int = 20,
                                 similarity_method: str = 'cosine') -> List[Tuple[str, float]]:
        
        rated_items = set(self.data.user_ratings.get(user_id, {}).keys())
        all_items = set(self.data.item_ids)
        unrated_items = all_items - rated_items
        
        predictions = []
        
        for item_id in unrated_items:
            if method == 'user_based':
                pred = self.predict_rating_user_based(
                    user_id, item_id, k_neighbors, similarity_method
                )
            elif method == 'item_based':
                pred = self.predict_rating_item_based(
                    user_id, item_id, k_neighbors, similarity_method
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            predictions.append((item_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    
    def precompute_similarities(self, compute_user: bool = True,
                               compute_item: bool = True,
                               max_users: Optional[int] = None,
                               max_items: Optional[int] = None):
        
        if compute_user:
            self.user_sim_matrix = self.similarity_calc.compute_user_similarity_matrix(
                self.data, max_users=max_users
            )
        
        if compute_item:
            self.item_sim_matrix = self.similarity_calc.compute_item_similarity_matrix(
                self.data, max_items=max_items
            )