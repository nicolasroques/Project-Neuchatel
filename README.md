# Detecting the level difficulty of French texts
## Team Neuchâtel
### by Francisco Díaz and Nicolas Roques

# About
The goal is to develop a model specifically for English speakers learning French. The purpose of this model is to assess the difficulty level of French texts, categorizing them according to the Common European Framework of Reference for Languages (CEFR) levels, ranging from A1 (beginner) to C2 (advanced). This tool would be useful in a recommendation system that suggests French texts, such as news articles, aligned with the learner's current proficiency level. For instance, if a user is at an A1 level in French, presenting them with a B2 level text would be overwhelming. The ideal text for language learning should predominantly contain familiar words, with just a few new or challenging terms to facilitate learning and skill improvement.

# Organisation of the code

All our code files contain models to classify the difficulty of French texts.

Our code is organised in three files: the first file contains the four models Logistic regression, kNN, Decision Tree and Random Forests as well as Logistic regression with data cleaning and Word embedding with SVC. The second file is composed of the Keras model. The third files contains the CamemBERT model. We have presented them in the next section.

In each file of our project, we followed a structured process. First, we loaded the dataset, which is used for both training and testing our models. Then, we imported the necessary packages to ensure we had all the essential tools. The third step involved preparing the data for classification, a key phase for accuracy in our models.

After preparing the data, we defined the methods for training and testing the models we were working on. This step was for creating a foundation for our analysis. Then, we moved on to predictions on the unlabelled test data. Finally, we evaluated the models' performance, ensuring they accurately classified the difficulty level of French texts. If they did not, we would go back to training and testing with other approaches or parameters. 



|                  | Logistic regression | kNN | Decision Tree | Random Forests |
|------------------|---------------------|-----|---------------|----------------|
| Precision        | 0.4645              | 0.4021 | 0.3066   | 0.4303         |
| Recall           | 0.4677              | 0.3194 | 0.3060   | 0.4202         |
| F1-score         | 0.4640              | 0.3029 | 0.3033   | 0.4074         |
| Accuracy         | 0.4667              | 0.3198   | 0.3063 | 0.4177         |


|                  | Logistic regression with data cleaning | Word embedding with SVC | Keras Sequential | CamemBERT |
|------------------|---------------------|-----|---------------|----------------|
| Precision        | 0.4418              | 0.5038 | 0.5280   | 0.6168 |
| Recall           | 0.4396              | 0.5054 | 0.5240   | 0.5917 |
| F1-score         | 0.4360              | 0.5035 | 0.5099   | 0.5941 |
| Accuracy         | 0.4385              | 0.5062 | 0.5208   | 0.5917 |

