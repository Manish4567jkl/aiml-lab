import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('Iris.csv')  
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].values   


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
       
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

     
        for _ in range(self.iterations):
            model = sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / self.m) * np.dot(X.T, (model - y))
            db = (1 / self.m) * np.sum(model - y)

       
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
 
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]


log_reg = LogisticRegression(learning_rate=0.01, iterations=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()