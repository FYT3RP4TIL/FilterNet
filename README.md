# 📧 FilterNet : Pinpoint Email Spam Detection

## 🔍 Overview
FilterNet is an advanced machine learning project designed to classify email messages as either spam or ham (non-spam). Utilizing state-of-the-art Natural Language Processing (NLP) techniques and a Naive Bayes classifier, FilterNet provides a robust solution for email filtering.

## ✨ Features
- 📊 Efficient data preprocessing and analysis
- 🔤 Advanced text vectorization using Bag of Words model
- 🧠 Naive Bayes classification for optimal spam detection
- 📈 Comprehensive model evaluation and visualization
- 🚀 Scalable pipeline for streamlined workflow

## 🧠 Methodology

### 1. Data Preprocessing
- Dataset: `spam.csv`
- Features: 'Message' (text) and 'Category' (spam/ham)
- Binary 'spam' column: 1 for spam, 0 for ham

### 2. Text Vectorization: Bag of Words
FilterNet employs the Bag of Words model for text-to-numerical feature conversion:

```python
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_cv = v.fit_transform(X_train.values)
```

- **Advantages**:
  - Simple and intuitive approach
  - Preserves word frequency information
- **Limitations**:
  - Loses word order context
  - Can result in large, sparse matrices

### 3. Classification Model: Naive Bayes
FilterNet utilizes the Multinomial Naive Bayes classifier:

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
```

- **Why Naive Bayes for FilterNet?**
  - Highly effective for text classification tasks
  - Performs well with high-dimensional data
  - Fast training and prediction capabilities
  - Efficient with relatively small datasets

- **Multinomial Naive Bayes in FilterNet**:
  - Tailored for discrete features (e.g., word counts)
  - Assumes features follow a simple multinomial distribution

### 4. FilterNet Pipeline
FilterNet leverages scikit-learn's Pipeline for a streamlined workflow:

```python
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
```

- **Benefits of FilterNet's Pipeline**:
  - Encapsulates multiple steps in a single, coherent object
  - Prevents data leakage during cross-validation
  - Simplifies application of consistent steps to training and test data

### 5. Model Evaluation in FilterNet
FilterNet employs various metrics for comprehensive model evaluation:

1. **Confusion Matrix**: Visualizes true positives, true negatives, false positives, and false negatives.
2. **ROC Curve**: Plots True Positive Rate against False Positive Rate.
3. **Precision-Recall Curve**: Particularly useful for FilterNet's potentially imbalanced dataset.
4. **Classification Report**: Provides detailed precision, recall, and F1-score for each class.
5. **Learning Curve**: Helps diagnose potential overfitting or underfitting in FilterNet.

## 📊 Results
Initial results from FilterNet show exceptional performance:

```
         precision    recall  f1-score   support
     0       0.99      1.00      0.99       966
     1       0.99      0.93      0.96       149
accuracy                           0.99      1115
macro avg    0.99      0.97      0.98      1115
weighted avg 0.99      0.99      0.99      1115
```

- **Interpretation of FilterNet's Performance**:
  - High precision and recall achieved for both spam and ham classes
  - Overall accuracy of 99%, demonstrating FilterNet's effectiveness
  - Slightly lower recall for spam (class 1) indicates potential for minor improvements in spam detection

## 📈 Visualization and Interpretation

FilterNet's performance is further illustrated through the following visualizations:

### 1. Confusion Matrix
![FilterNet Confusion Matrix](https://github.com/user-attachments/assets/017ec1ee-bffc-4c11-883f-e2efe0a7335a)

- **Interpretation**: 
  - Provides a tabular summary of FilterNet's predictions vs. actual values
  - Diagonal elements represent correct predictions; off-diagonal elements are misclassifications
  - High concentration on the diagonal indicates FilterNet's strong performance

### 2. ROC Curve
![FilterNet ROC Curve](https://github.com/user-attachments/assets/d74c4653-7304-44a5-8b29-d340229f3de5)

- **Interpretation**:
  - Plots FilterNet's True Positive Rate against False Positive Rate
  - Area Under the Curve (AUC) quantifies FilterNet's class distinction ability
  - AUC close to 1.0 suggests excellent classification performance
  - Curve's proximity to top-left corner indicates FilterNet's effectiveness

### 3. Precision-Recall Curve
![FilterNet Precision-Recall Curve](https://github.com/user-attachments/assets/4be32d55-f3da-4c3e-ab6c-205e74ad8b8a)

- **Interpretation**:
  - Illustrates the tradeoff between precision and recall for different FilterNet thresholds
  - Particularly insightful for FilterNet's potentially imbalanced dataset
  - Area under this curve (Average Precision) provides a concise performance summary
  - High curve for both precision and recall indicates FilterNet's strong performance

### 4. Classification Report as a Heatmap
![FilterNet Classification Report Heatmap](https://github.com/user-attachments/assets/f5bc8372-a5d4-40a2-8c86-ebe18e69626e)

- **Interpretation**:
  - Visually represents FilterNet's precision, recall, and F1-score for each class
  - Darker colors typically indicate better performance
  - Allows quick identification of any class-specific issues in FilterNet's performance

### 5. Learning Curve
![FilterNet Learning Curve](https://github.com/user-attachments/assets/93e33ed2-fade-4817-8c9c-1cc4eb2281e0)

- **Interpretation**:
  - Shows FilterNet's performance on training and validation sets as training size increases
  - Helps diagnose overfitting or underfitting in FilterNet:
    - Large gap between training and validation scores may indicate overfitting
    - Low, close scores might suggest underfitting
  - Expectation: validation score increases and plateaus as training size grows

### Key Takeaways from FilterNet Visualizations
1. Confusion matrix and ROC curve confirm FilterNet's high accuracy reported in the classification report.
2. Precision-recall curve demonstrates FilterNet's ability to maintain high precision at high recall levels, crucial for effective spam detection.
3. Heatmap provides an easy-to-read overview of FilterNet's performance across different metrics and classes.
4. Learning curve suggests FilterNet is neither overfitting nor underfitting, and uses an appropriate amount of training data.

These visualizations offer a comprehensive view of FilterNet's performance, confirming its effectiveness in distinguishing between spam and ham messages across various evaluation metrics.

## 🚀 Future Improvements for FilterNet
1. **Advanced Feature Engineering**:
   - Explore TF-IDF vectorization for more nuanced text representation
   - Implement n-grams to capture phrase patterns and word sequences

2. **Model Expansion**:
   - Compare FilterNet's performance with other algorithms (e.g., SVM, Random Forest)
   - Implement ensemble methods for potentially higher accuracy

3. **Hyperparameter Optimization**:
   - Utilize Grid Search or Random Search to fine-tune FilterNet's parameters

4. **Cutting-edge NLP Techniques**:
   - Incorporate advanced text preprocessing like lemmatization
   - Explore word embeddings (e.g., Word2Vec, GloVe) for more sophisticated text representation

5. **Addressing Class Imbalance**:
   - Investigate techniques like SMOTE to balance spam and ham classes

6. **In-depth Error Analysis**:
   - Conduct thorough analysis of FilterNet's misclassifications to identify improvement areas

## 📚 References
- Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- Naive Bayes and Text Classification: [http://www.cs.cornell.edu/home/llee/papers/sentiment.pdf](http://www.cs.cornell.edu/home/llee/papers/sentiment.pdf)
- Introduction to Information Retrieval (Manning, Raghavan, Schütze): [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

---

<p align="center">
  FilterNet: Powering Intelligent Email Classification
</p>
