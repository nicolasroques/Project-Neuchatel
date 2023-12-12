# Detecting the level difficulty of French texts

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

