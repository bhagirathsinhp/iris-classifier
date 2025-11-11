# 🌸 Iris Flower Classification using Decision Tree

This project demonstrates a simple yet effective **machine learning workflow** using the classic **Iris dataset**.  
The goal is to classify iris flowers into three species: *setosa*, *versicolor*, and *virginica*, using a **Decision Tree Classifier** from `scikit-learn`.

---

## 📚 Dataset

The project uses the built-in **Iris dataset** from `sklearn.datasets`.

- **Samples:** 150
- **Features (4):**
  - sepal length (cm)
  - sepal width (cm)
  - petal length (cm)
  - petal width (cm)
- **Target classes:**  
  - 0 → setosa  
  - 1 → versicolor  
  - 2 → virginica  

---

## 🧠 Model Used

- **Decision Tree Classifier**
  - `random_state=42` (for reproducibility)

---

## 🛠 Steps Performed

1. Load the Iris dataset
2. Split the data into training and testing sets (80% train, 20% test)
3. Train the Decision Tree classifier
4. Predict the classes for the test set
5. Evaluate the model using:
    - Accuracy score
    - Confusion matrix
6. Export results automatically:
    - Saves the confusion matrix as a PNG image
    - Saves the trained Decision Tree model as a .joblib file

All exported files are stored in an outputs/ folder.

---

## 📦 Requirements

- Make sure you have the following libraries installed:

      scikit-learn>=1.3
      matplotlib>=3.8
      seaborn>=0.13
      jupyter>=1.0
      pytest>=8.0
      joblib>=1.3

---

## 📂 Project Structure

    ┣ notebooks/
    ┃ ┗ iris_model.ipynb
    ┣ outputs/
    ┃ ┣ confusion_matrix.png
    ┃ ┗ decision_tree_model.joblib
    ┣ src/
    ┃ ┗ train.py
    ┣ tests/
    ┃ ┗ test_train.py
    ┣ .gitignore
    ┣ README.md
    ┗ requirements.txt

---

## ✅ Results

- After running train.py, the following will be generated automatically:
  ✅ outputs/confusion_matrix.png — Heatmap image of confusion matrix
  ✅ outputs/decision_tree_model.joblib — Trained model file

- Additionally:
  - First few predictions and their corresponding true labels are printed
  - Full confusion matrix is displayed in the console
  - Overall model accuracy is printed

---

## ▶️ How to Run

### 🧩 Set Up a Python Virtual Environment

**Windows:**

    python -m venv venv

**macOS / Linux:**

    source venv/bin/activate

**Install dependencies:**

    pip install -r requirements.txt

**Run the script:**

    python train.py

**OR run interactively in Jupyter Notebook:**
  - Open notebook.ipynb
  - Each code snippet is placed in separate cells for easy execution and understanding

---

## 👤 Author

[![GitHub: bhagirathsinhp](https://img.shields.io/github/followers/bhagirathsinhp?label=Follow&style=social)](https://github.com/bhagirathsinhp)
[![LinkedIn: Bhagirath Parmar](https://img.shields.io/badge/-Bhagirath%20Parmar-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/bhagirath-parmar-385865269/)](https://www.linkedin.com/in/bhagirath-parmar-385865269/)
