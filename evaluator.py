import numpy as np
from typing import Dict, List, Tuple
from data_manager import DataManager
from similarity import SimilarityCalculator
from recommender import CollaborativeFiltering

class Evaluator:

    
    @staticmethod
    def mean_absolute_error(predictions: List[float], actuals: List[float]) -> float:
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        return np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    
    @staticmethod
    def root_mean_square_error(predictions: List[float], actuals: List[float]) -> float:
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        return np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
    
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:

        top_k = recommended[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in top_k if item in relevant_set)
        return hits / k
    
    def evaluate(self, train_data: DataManager, test_data: DataManager,
                method: str = 'user_based', k_neighbors: int = 20,
                similarity_method: str = 'cosine') -> Dict[str, float]:

        print(f"\nEvaluating {method} CF with {similarity_method} similarity...")
        
        similarity_calc = SimilarityCalculator()
        recommender = CollaborativeFiltering(train_data, similarity_calc)
        
        predictions = []
        actuals = []
        
        test_count = 0
        for user_id in test_data.user_ratings:
            for item_id, actual_rating in test_data.user_ratings[user_id].items():

                if user_id not in train_data.user_ratings:
                    continue
                if item_id not in train_data.item_ratings:
                    continue
                

                if method == 'user_based':
                    pred_rating = recommender.predict_rating_user_based(
                        user_id, item_id, k_neighbors, similarity_method
                    )
                else:  
                    pred_rating = recommender.predict_rating_item_based(
                        user_id, item_id, k_neighbors, similarity_method
                    )
                
                predictions.append(pred_rating)
                actuals.append(actual_rating)
                test_count += 1
        
        print(f"Evaluated on {test_count} test ratings")
        
        mae = self.mean_absolute_error(predictions, actuals)
        rmse = self.root_mean_square_error(predictions, actuals)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'test_size': test_count
        }
    
    def compare_methods(self, train_data: DataManager, test_data: DataManager,
                       k_values: List[int] = [5, 10, 20, 50],
                       similarity_methods: List[str] = ['cosine', 'pearson']):
        
        
        results = []
        
        for method in ['user_based', 'item_based']:
            for sim_method in similarity_methods:
                for k in k_values:
                    print(f"\nTesting: {method}, similarity={sim_method}, k={k}")
                    
                    metrics = self.evaluate(
                        train_data, test_data, 
                        method=method,
                        k_neighbors=k,
                        similarity_method=sim_method
                    )
                    
                    results.append({
                        'method': method,
                        'similarity': sim_method,
                        'k': k,
                        **metrics
                    })
        
        return results

