
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the student dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Select the input columns
x = data[["studytime", "failures", "absences", "G1", "G2"]]

# Select the final grade as the output
y = data["G3"]

# Divide the data into training and testing parts
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create and train the machine learning model
model = LinearRegression()
model.fit(x_train, y_train)

# Test the model
predictions = model.predict(x_test)
error = mean_absolute_error(y_test, predictions)

print("STUDENT PERFORMANCE PREDICTION")
print("--------------------------------")
print("Model training completed successfully!")
print("Average prediction error:", round(error, 2), "marks out of 20")

# Get information about a new student
print("\nEnter the student's details:")

studytime = int(input("Study time (1-4): "))
failures = int(input("Past class failures (0-3): "))
absences = int(input("Number of school absences: "))
g1 = float(input("First period grade (0-20): "))
g2 = float(input("Second period grade (0-20): "))

# Prepare the student's data
student = pd.DataFrame(
    [[studytime, failures, absences, g1, g2]],
    columns=["studytime", "failures", "absences", "G1", "G2"]
)

# Predict the final grade
result = model.predict(student)[0]

# Keep the displayed prediction between 0 and 20
result = max(0, min(20, result))

print("\n--------------------------------")
print("Predicted final mathematics grade:", round(result, 2), "/ 20")
print("--------------------------------")
