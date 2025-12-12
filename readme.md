# Collaborative Filtering Recommender System

## 📋 Project Overview
A comprehensive implementation of collaborative filtering recommendation algorithms comparing user-based and item-based approaches. This system predicts user ratings and generates personalized recommendations using cosine similarity and Pearson correlation.

## 👥 Team Members
- **Student 1**: Data & Similarity Foundation
- **Student 2**: Core Algorithm Engineer  
- **Student 3**: Evaluation Specialist
- **Student 4**: UI/UX Developer
- **Student 5**: Analysis & Comparison Expert

## 🚀 Quick Start

### Installation
```bash
# Install required packages
pip install numpy pandas matplotlib

# Verify installation
python test_install.py
```

### Run the System
```bash
# Main recommender system with interactive mode
python main.py

# Algorithm comparison on patterned datasets
python compare_all_patterns.py

# Generate synthetic test datasets
python create_patterned_data.py
```

## 📁 File Structure

```
├── core/
│   ├── data_manager.py      # Data loading and preprocessing
│   ├── similarity.py        # Similarity calculations
│   ├── recommender.py       # CF algorithms implementation
│   └── evaluator.py         # Performance evaluation
├── main/
│   ├── main.py              # Main program with UI
│   └── compare_all_patterns.py  # Comparative analysis
├── data/
│   └── create_patterned_data.py  # Dataset generation
└── test_install.py          # Environment verification
```

## 📊 Features

### Core Algorithms
- **User-Based Collaborative Filtering**: Predicts ratings based on similar users
- **Item-Based Collaborative Filtering**: Predicts ratings based on similar items
- **Similarity Metrics**: Cosine similarity and Pearson correlation
- **Top-N Recommendations**: Generate personalized item rankings

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Emphasizes larger errors
- **Precision@K**: Accuracy of top-K recommendations

### Dataset Support
- Sample synthetic datasets
- MovieLens compatibility (if available)
- Patterned datasets for algorithm testing
- Custom CSV/JSON data formats

## 🎮 Interactive Mode Commands

```
recommend <user_id> <method> <n>  # Get top N recommendations
predict <user_id> <item_id> <method>  # Predict specific rating
users  # List all users
items  # List all items
stats  # Show dataset statistics
quit   # Exit interactive mode
```

## 📈 Generated Outputs

### Data Files
- `sample_ratings.csv` - Sample dataset
- `patterned_data_*.csv` - Synthetic datasets
- `current_ratings.csv` - Active dataset

### Results & Visualizations
- `evaluation_results.png` - Performance charts
- `clear_comparison_results.png` - Algorithm comparisons
- `patterned_datasets_comparison.csv` - Evaluation results

## 🔧 Dependencies

```python
numpy      # Numerical computations
pandas     # Data manipulation  
matplotlib # Data visualization
```

## 📚 Usage Examples

### Basic Recommendation
```bash
python main.py
# Follow interactive prompts
# > recommend U1 user_based 5
# > predict U1 I10 item_based
```

### Full Algorithm Comparison
```bash
python create_patterned_data.py
python compare_all_patterns.py
```

### Custom Dataset
1. Prepare CSV with columns: `user_id`, `item_id`, `rating`
2. Update paths in `main.py`
3. Run: `python main.py`

## 🧪 Testing

```bash
# Quick functionality test
python main.py  # Select "y" when prompted for quick test

# Generate test datasets
python create_patterned_data.py

# Run comprehensive comparisons
python compare_all_patterns.py
```

## 📝 Implementation Details

### Data Management
- Sparse matrix representation for efficiency
- Train/test split with configurable ratios
- Support for multiple data formats
- User/item mapping and indexing

### Similarity Calculations
- Cosine similarity: $ \text{cos}(θ) = \frac{A·B}{||A||·||B||} $
- Pearson correlation: $ r = \frac{\sum{(x_i-\bar{x})(y_i-\bar{y})}}{\sqrt{\sum{(x_i-\bar{x})^2}\sum{(y_i-\bar{y})^2}}} $
- Matrix precomputation for performance

### Prediction Formulas
- **User-Based**: $ r_{u,i} = \bar{r}_u + \frac{\sum sim(u,v)·(r_{v,i}-\bar{r}_v)}{\sum |sim(u,v)|} $
- **Item-Based**: $ r_{u,i} = \bar{r}_i + \frac{\sum sim(i,j)·(r_{u,j}-\bar{r}_j)}{\sum |sim(i,j)|} $

## 🎯 Performance Optimization

- Sparse data structures for memory efficiency
- Similarity matrix caching
- Configurable neighborhood sizes (k)
- Batch processing for evaluations
