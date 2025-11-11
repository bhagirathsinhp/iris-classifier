import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Ensure outputs/ folder exists
os.makedirs("outputs", exist_ok=True)

iris = load_iris()
model = DecisionTreeClassifier(random_state=42)

x = iris.data  # shape (150, 4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Prediction:", y_pred[:5])
print("True labels:", y_test[:5])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print(confusion)

# --- SAVE MODEL ---
joblib.dump(model, "outputs/decision_tree_model.joblib")

# --- SAVE CONFUSION MATRIX PLOT ---
plt.figure(figsize=(4,4))
plt.imshow(confusion, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(len(confusion)):
    for j in range(len(confusion[0])):
        plt.text(j, i, confusion[i, j], ha='center', va='center', color='black')

plt.savefig("outputs/confusion_matrix.png")
plt.close()
