from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
model = DecisionTreeClassifier(random_state=42)

x = iris.data  #shape (150, 4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Prediction:", y_pred[:5])
print("True labels:", y_pred[:5])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print(confusion)