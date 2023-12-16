# 📜 Detecting the difficulty level of French texts
## 🏙️ Team Neuchâtel
### 🙋‍♂️🙋 by Francisco Díaz and Nicolas Roques

**Link to our video**:

# 💡 About
The goal is to develop a model specifically for English speakers learning French. The purpose of this model is to assess the difficulty level of French texts, categorizing them according to the Common European Framework of Reference for Languages (CEFR) levels, ranging from A1 (beginner) to C2 (advanced). This tool would be useful in a recommendation system that suggests French texts, such as news articles, aligned with the learner's current proficiency level. For instance, if a user is at an A1 level in French, presenting them with a B2 level text would be overwhelming. The ideal text for language learning should predominantly contain familiar words, with just a few new or challenging terms to facilitate learning and skill improvement.

# 🗂️ Organisation of our code

All our code files contain models to classify the difficulty of French texts.

Our code is organised in three files: the first file contains the four models Logistic regression, kNN, Decision Tree and Random Forests as well as Logistic regression with data cleaning and Word embedding with SVC. The second file contains the CamemBERT model. We have presented them in the next section.

In each file of our project, we followed a structured process. First, we loaded the dataset, which is used for both training and testing our models. Then, we imported the necessary packages to ensure we had all the essential tools. The third step involved preparing the data for classification, a key phase for accuracy in our models.

After preparing the data, we defined the methods for training and testing the models we were working on. This step was for creating a foundation for our analysis. Then, we moved on to predictions on the test set we created with the full training data. Finally, we evaluated the models' performance, ensuring they accurately classified the difficulty level of French texts. If they did not, we would go back to training and testing with other approaches or parameters. 

# </> Presentation of our code for each model
For each model, we evaluated the model using precision, recall, F1-score, and accuracy.

1️⃣ **Logistic regression**
- Used TF-IDF vectorization to convert text data into TF-IDF features.
- Applied Logistic Regression as the classifier.
- Split the dataset into training and testing sets.
- Made predictions and prepared the submission.

2️⃣ **kNN**
- Implemented kNN classifier with TF-IDF vectorization.
- Conducted hyperparameter tuning using GridSearchCV to find the best 'k' value.

3️⃣ **Decision Tree**
- Employed a Decision Tree classifier with TF-IDF vectorization.
  
4️⃣ **Random Forests**
- Utilized Random Forest Classifier with TF-IDF vectorization.

5️⃣ **Logistic regression with data cleaning**
- Added data cleaning steps, including lowercasing, removing punctuation, and stopwords.
- Re-applied logistic regression with the cleaned data.
- Conducted hyperparameter tuning for the TF-IDF vectorizer and logistic regression.

6️⃣ **Word embedding with SVC**
- Implemented word embeddings using spaCy's French language model.
- Used different kernels of Support Vector Classifier (SVC) including RBF, Linear, and Sigmoid.
- Conducted hyperparameter tuning with GridSearchCV.
- Evaluated the models and selected the best-performing kernel.

7️⃣ **CamemBERT**
- Used the CamemBERT model, a language model specifically trained on French language, fine-tuned on the training dataset


# 📊 Our results
|                  | Logistic regression | kNN | Decision Tree | Random Forests |
|------------------|---------------------|-----|---------------|----------------|
| Precision        | 0.4645              | 0.4021 | 0.3066   | 0.4303         |
| Recall           | 0.4677              | 0.3194 | 0.3060   | 0.4202         |
| F1-score         | 0.4640              | 0.3029 | 0.3033   | 0.4074         |
| Accuracy         | 0.4667              | 0.3198   | 0.3063 | 0.4177         |

|                  | Logistic regression with data cleaning |Ridge regression  | Word embedding with SVC  | CamemBERT |
|------------------|---------------------|-----|---------|------------------------------|
| Precision        | 0.4418              |0.4739         | 0.5038  |  0.6079|
| Recall           | 0.4396              |0.4752         | 0.5054  | 0.5875 |
| F1-score         | 0.4360              |0.4709         | 0.5035  | 0.5941 |
| Accuracy         | 0.4385              |0.4740         | 0.5062  | 0.5883 |

# 💻 Presentation of our Streamlit application

1️⃣ **Difficulty Prediction**: Predicts the difficulty level of French sentences.

2️⃣ **Text Complexity Analysis**: Calculates metrics like average word length and readability score.

3️⃣ **Translation and Audio Pronunciation**: Translates French to English and provides audio pronunciation of sentences.

4️⃣ **Part-of-Speech Tagging**: Identifies grammatical components of sentences.

5️⃣ **Interactive Filters and Sorting**: Allows users to filter and sort text based on difficulty level and other criteria.

6️⃣ **Data Visualization**: Offers visual charts to display text analysis results.

7️⃣ **Export Functionality**: Enables exporting of analyzed data for further use.


