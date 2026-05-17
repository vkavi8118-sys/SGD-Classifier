# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start 2.Import libraries: pandas, numpy, sklearn.datasets, sklearn.model_selection, sklearn.preprocessing, sklearn.linear_model, sklearn.metrics. 3.Load dataset: Use load_iris() from sklearn.datasets. It contains 150 samples of 3 classes: Setosa, Versicolor, Virginica. 4.Prepare features and target: a.Features X = sepal length, sepal width, petal length, petal width. b.Target y = species label.

5.Split dataset: Divide into training and testing sets (80%-20%). 6.Feature scaling: Standardize the features using StandardScaler (SGD converges better with normalized data). 7.Initialize SGD Classifier: Use SGDClassifier(loss="log_loss") for logistic regression-like classification.

8.Train model: Fit the classifier on training data. 9.Predict species: Use the trained model to predict species on test data. 10.Evaluate performance: a.Accuracy score. b.Confusion matrix. c.Classification report.


## Program:
```
/*

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build and train SGD Classifier
model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print(" Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=iris.target_names, 
                     columns=iris.target_names)
print("\nConfusion Matrix:\n", cm_df)

report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:\n", report_df.round(2))
Developed by:Kavinaya V 
RegisterNumber:212225230133
*/
```

## Output:
<img width="550" height="340" alt="ml 7" src="https://github.com/user-attachments/assets/c73f77f0-2f46-4ce1-95a6-1100f5b994c0" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
