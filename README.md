
# Flipkart Sentiment Analysis

A machine learning project that analyzes customer reviews from Flipkart to predict sentiment using Natural Language Processing (NLP) techniques and Decision Tree classification.

# 👨‍💻 Developer Information 
- **Developer**: P.Manasa
- **Roll No**: 222T1A3145
- **Institution**: Ashoka Womens Engineering College.

## 🎯 Project Overview

This project performs sentiment analysis on Flipkart customer reviews to classify them as positive or negative. It uses text preprocessing, TF-IDF vectorization, and a Decision Tree classifier to predict sentiment based on review content.

## 🚀 Features

- **Data Preprocessing**: Cleans and preprocesses review text by removing stopwords and converting to lowercase
- **Sentiment Labeling**: Automatically labels reviews as positive (rating ≥ 4) or negative (rating < 4)
- **Text Vectorization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numerical features
- **Machine Learning**: Implements Decision Tree classifier for sentiment prediction
- **Data Visualization**: 
  - Sentiment distribution bar chart
  - Word cloud for positive reviews
  - Confusion matrix heatmap
- **Performance Metrics**: Calculates accuracy and displays confusion matrix

## 📋 Requirements

```
pandas
nltk
scikit-learn
matplotlib
seaborn
wordcloud
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis
```

2. Install required packages:
```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud
```

3. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## 📊 Dataset

The project uses a Flipkart reviews dataset (`flipkart.csv`) containing:
- `review`: Customer review text
- `rating`: Product rating (used to determine sentiment)

**Dataset Requirements:**
- CSV format with columns: 'review', 'rating'
- Place the dataset file as `flipkart.csv` in your project directory

## 🔧 Usage

1. Update the file path in the script:
```python
file_path = r'path/to/your/flipkart.csv'
```

2. Run the analysis:
```python
python sentiment_analysis.py
```

## 📈 Methodology

### 1. Data Preprocessing
- Convert reviews to lowercase
- Remove English stopwords using NLTK
- Create binary sentiment labels (1 for positive, 0 for negative)

### 2. Feature Engineering
- TF-IDF Vectorization with maximum 5000 features
- Converts text reviews into numerical feature vectors

### 3. Model Training
- Split data into 80% training and 20% testing
- Train Decision Tree Classifier
- Use random_state=42 for reproducibility

### 4. Evaluation
- Calculate accuracy score
- Generate confusion matrix
- Visualize results with heatmap

## 📊 Visualizations

The project generates three key visualizations:

1. **Sentiment Distribution**: Bar chart showing the balance between positive and negative reviews
2. **Word Cloud**: Visual representation of most frequent words in positive reviews
3. **Confusion Matrix**: Heatmap showing model performance metrics

## 🎯 Model Performance

The Decision Tree classifier's performance is evaluated using:
- **Accuracy Score**: Overall percentage of correctly classified reviews
- **Confusion Matrix**: Detailed breakdown of true positives, true negatives, false positives, and false negatives

## 🔍 Key Functions

### `preprocess_reviews_stopwords(df)`
- Preprocesses review text and creates sentiment labels
- Parameters: DataFrame with 'review' and 'rating' columns
- Returns: Cleaned DataFrame with sentiment column

## 🚀 Future Enhancements

- [ ] Implement additional ML algorithms (Random Forest, SVM, Neural Networks)
- [ ] Add cross-validation for better model evaluation
- [ ] Include feature importance analysis
- [ ] Implement hyperparameter tuning
- [ ] Add support for multi-class sentiment (neutral, positive, negative)
- [ ] Create a web interface for real-time sentiment prediction
- [ ] Add more sophisticated text preprocessing (stemming, lemmatization)

## 📝 File Structure

```
flipkart-sentiment-analysis/
│
├── sentiment_analysis.py          # Main analysis script
├── flipkart.csv                  # Dataset file
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```




## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- Scikit-learn for machine learning algorithms
- Matplotlib and Seaborn for data visualization
- WordCloud library for text visualization


