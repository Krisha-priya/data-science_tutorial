import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Create a Synthetic Dataset
np.random.seed(42)  # for reproducibility

n_samples = 200
study_hours = np.random.uniform(10, 50, n_samples)
practice_tests = np.random.randint(0, 10, n_samples)
gpa = np.random.uniform(2.5, 4.0, n_samples)

# Create a binary outcome: Passed (1) or Failed (0)
# Students with more study hours, practice tests, and higher GPA are more likely to pass.
passed = (study_hours * 0.5 + practice_tests * 1.5 + (gpa - 2.5) * 5 + np.random.normal(0, 5, n_samples) > 20).astype(int)

df = pd.DataFrame({'study_hours': study_hours, 'practice_tests': practice_tests, 'gpa': gpa, 'passed': passed})


# 2. Explain the Dataset
print("Dataset Description:")
print(df.describe())
print("\nFirst few rows:")
print(df.head())

# 3. Prepare the Data
X = df[['study_hours', 'practice_tests', 'gpa']]  # Features
y = df['passed']  # Target variable

# 4. Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 7. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# 8. Predict for Unknown Students (VPs in this context)
print("\nPredictions for Unknown Students:")
unknown_students = pd.DataFrame({
    'study_hours': [35, 20, 45, 15],  # Example study hours
    'practice_tests': [7, 2, 9, 1],  # Example practice tests
    'gpa': [3.7, 2.8, 3.9, 2.6]       # Example GPAs
})

predictions = model.predict(unknown_students)
print(unknown_students)
print("Predicted pass/fail status:", predictions)

probabilities = model.predict_proba(unknown_students)
print("Predicted probabilities:", probabilities)
