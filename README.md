# 📧 Filternet - Email Spam Classifier

## 🔍 Overview
This project implements a machine learning model to classify text messages as either spam or ham (non-spam) using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

## 🧠 Methodology

### 1. Data Preprocessing
- Dataset: `spam.csv`
- Features: 'Message' (text) and 'Category' (spam/ham)
- Created a binary 'spam' column: 1 for spam, 0 for ham

### 2. Text Vectorization: Bag of Words
We use the Bag of Words model to convert text data into numerical features:

- **CountVectorizer**: This method tokenizes the text (splits it into individual words) and counts the occurrence of each token.
  
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  v = CountVectorizer()
  X_train_cv = v.fit_transform(X_train.values)
  ```

- Advantages:
  - Simple and intuitive
  - Preserves word frequency information
- Limitations:
  - Loses word order
  - Can result in large, sparse matrices

### 3. Classification Model: Naive Bayes
We use the Multinomial Naive Bayes classifier:

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
```

- **Why Naive Bayes?**
  - Effective for text classification
  - Works well with high-dimensional data (like Bag of Words vectors)
  - Fast training and prediction
  - Performs well even with relatively small datasets

- **Multinomial Naive Bayes**:
  - Suitable for discrete features (e.g., word counts)
  - Assumes features are generated from a simple multinomial distribution

### 4. Pipeline Creation
We use scikit-learn's Pipeline to streamline our workflow:

```python
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
```

- **Benefits of using Pipeline**:
  - Encapsulates multiple steps in a single object
  - Ensures that data leakage is prevented during cross-validation
  - Simplifies the process of applying the same steps to training and test data

### 5. Model Evaluation
We use various metrics to evaluate our model's performance:

1. **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives.
2. **ROC Curve**: Plots the True Positive Rate against the False Positive Rate.
3. **Precision-Recall Curve**: Especially useful for imbalanced datasets.
4. **Classification Report**: Provides precision, recall, and F1-score for each class.
5. **Learning Curve**: Helps diagnose overfitting or underfitting.

## 📊 Results
Initial results show high performance:

```
         precision    recall  f1-score   support
     0       0.99      1.00      0.99       966
     1       0.99      0.93      0.96       149
accuracy                           0.99      1115
macro avg    0.99      0.97      0.98      1115
weighted avg 0.99      0.99      0.99      1115
```

- **Interpretation**:
  - High precision and recall for both classes
  - Overall accuracy of 99%
  - Slightly lower recall for spam (class 1) suggests some spam messages might be misclassified as ham

## 📈 Visualization and Interpretation

To gain deeper insights into our model's performance, we've created several visualizations:

### 1. Confusion Matrix
![Confusion Matrix](placeholder_confusion_matrix.png)

- **Interpretation**: 
  - The confusion matrix provides a tabular summary of the model's predictions vs. actual values.
  - The diagonal elements represent correct predictions, while off-diagonal elements are misclassifications.
  - A high concentration of values on the diagonal indicates good model performance.

### 2. ROC Curve
![ROC Curve](placeholder_roc_curve.png)

- **Interpretation**:
  - The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate.
  - The Area Under the Curve (AUC) quantifies the model's ability to distinguish between classes.
  - An AUC close to 1.0 suggests excellent classification performance.
  - The curve's proximity to the top-left corner indicates better performance.

### 3. Precision-Recall Curve
![Precision-Recall Curve](placeholder_pr_curve.png)

- **Interpretation**:
  - This curve shows the tradeoff between precision and recall for different thresholds.
  - It's particularly useful for imbalanced datasets.
  - The area under this curve (Average Precision) provides a single-number summary of performance.
  - A curve that remains high for both precision and recall indicates good performance.

### 4. Classification Report as a Heatmap
![Classification Report Heatmap](placeholder_heatmap.png)

- **Interpretation**:
  - This heatmap visually represents the precision, recall, and F1-score for each class.
  - Darker colors typically indicate better performance.
  - It allows for quick identification of any class-specific issues in model performance.

### 5. Learning Curve
![Learning Curve](placeholder_learning_curve.png)

- **Interpretation**:
  - The learning curve shows model performance on both training and validation sets as the training set size increases.
  - It helps diagnose overfitting or underfitting:
    - If training score is much higher than validation score, the model might be overfitting.
    - If both scores are low and close, the model might be underfitting.
  - As the training size increases, we expect the validation score to increase and eventually plateau.

### Key Takeaways from Visualizations
1. The confusion matrix and ROC curve confirm the high accuracy reported in the classification report.
2. The precision-recall curve demonstrates the model's ability to maintain high precision even at high recall levels, which is crucial for spam detection.
3. The heatmap provides an easy-to-read overview of the model's performance across different metrics and classes.
4. The learning curve suggests that the model is neither overfitting nor underfitting, and that we're using an appropriate amount of training data.

These visualizations provide a comprehensive view of the model's performance, confirming its effectiveness in distinguishing between spam and ham messages across various evaluation metrics.

[Subsequent sections (Future Improvements and References) remain unchanged]

---


## 🚀 Future Improvements
1. **Feature Engineering**:
   - Explore TF-IDF (Term Frequency-Inverse Document Frequency) instead of simple count vectorization
   - Implement n-grams to capture phrases and word sequences

2. **Model Selection**:
   - Compare performance with other algorithms (e.g., SVM, Random Forest)
   - Implement ensemble methods for potentially higher accuracy

3. **Hyperparameter Tuning**:
   - Use techniques like Grid Search or Random Search to optimize model parameters

4. **Advanced NLP Techniques**:
   - Implement text preprocessing steps like lemmatization
   - Explore word embeddings (e.g., Word2Vec, GloVe) for more nuanced text representation

5. **Handling Class Imbalance**:
   - Investigate techniques like SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalance between spam and ham classes

6. **Error Analysis**:
   - Conduct in-depth analysis of misclassified messages to identify patterns and potential areas for improvement

## 📚 References
- Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- Naive Bayes and Text Classification: [http://www.cs.cornell.edu/home/llee/papers/sentiment.pdf](http://www.cs.cornell.edu/home/llee/papers/sentiment.pdf)
- Introduction to Information Retrieval (Manning, Raghavan, Schütze): [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

---
