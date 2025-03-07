import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.preprocessing import LabelEncoder 

df = pd.read_csv("Task 1/Titanic-Dataset.csv")

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"]) 
df["Embarked"] = encoder.fit_transform(df["Embarked"]) 

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  

X = df.drop(columns=['Survived']) 
y = df["Survived"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy: {accuracy:.2f}") 

print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Purples",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()