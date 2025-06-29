#Importing necessary libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Importing and loading dataset
data = pd.read_csv('reviews.csv')
data.head()

#Defining features(X) and targets(y)
X = data['review_text']
y = data['label']

#Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y, shuffle=True)

#Initializing vectorizer and vectorizing training data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Initializing logistic regression model
model = LogisticRegression()
model.fit(X_train_vec,y_train)

#Getting predictions from the model
y_pred = model.predict(X_test_vec)

#Evaluating the model from y_test and y_pred
accuracy = accuracy_score(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)
con_mat = confusion_matrix(y_test, y_pred)
print(f"Accuracy Score of the model: {accuracy:.2f}")
print(f"classification Report of the model: {class_rep}")

#Confusion Matrix
sns.heatmap(con_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Genuine', 'Predicted Fake'],
            yticklabels=['Actual Genuine', 'Actual Fake'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#User Input
user_input = input("Enter your email 📧:\n")

vectorize_user_input = vectorizer.transform([user_input])

prediction = model.predict(vectorize_user_input)

output = "FAKE ❌" if prediction[0] == 'fake' else "Geniune ✅"
print(f"User commented review:\n{user_input}\n")
print(f"The review is classified as:\n{output}")