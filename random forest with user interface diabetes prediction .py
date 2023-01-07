#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout

class DiabetesPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

# Split the dataset into features and target
        data = pd.read_csv("pima-indians-diabetes.csv")
        data.head()
        data = data.rename(index=str, columns={"6":"preg"})
        data = data.rename(index=str, columns={"148":"gluco"})
        data = data.rename(index=str, columns={"72":"bp"})
        data = data.rename(index=str, columns={"35":"stinmm"})
        data = data.rename(index=str, columns={"0":"insulin"})
        data = data.rename(index=str, columns={"33.6":"mass"})
        data = data.rename(index=str, columns={"0.627":"dpf"})
        data = data.rename(index=str, columns={"50":"age"})
        data = data.rename(index=str, columns={"1":"target"})
        data.head()
        X = data.iloc[:, :-1]
        y = data.iloc[:,8]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create the random forest classifier
        self.clf = RandomForestClassifier()

        # Train the model on the training data
        self.clf.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = self.clf.predict(X_test)

        # Print the accuracy of the model
        accuracy = self.clf.score(X_test, y_test)
        print("Accuracy:", accuracy)
    def initUI(self):
        # Create the input widgets
        self.pregnancy_label = QLabel("Number of pregnancies:")
        self.pregnancy_input = QLineEdit()
        self.glucose_label = QLabel("Plasma glucose concentration:")
        self.glucose_input = QLineEdit()
        self.blood_pressure_label = QLabel("Diastolic blood pressure (mm Hg):")
        self.blood_pressure_input = QLineEdit()
        self.skin_thickness_label = QLabel("Triceps skin fold thickness (mm):")
        self.skin_thickness_input = QLineEdit()
        self.insulin_label = QLabel("2-Hour serum insulin (mu U/ml):")
        self.insulin_input = QLineEdit()
        self.bmi_label = QLabel("Body mass index (weight in kg/(height in m)^2):")
        self.bmi_input = QLineEdit()
        self.diabetes_pedigree_function_label = QLabel("Diabetes pedigree function:")
        self.diabetes_pedigree_function_input = QLineEdit()
        self.age_label = QLabel("Age (years):")
        self.age_input = QLineEdit()

        # Create the "Predict" button
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)

        # Create the output label
        self.output_label = QLabel()

        # Create a vertical layout to hold the input widgets and the output label
        vbox = QVBoxLayout()
        vbox.addWidget(self.pregnancy_label)
        vbox.addWidget(self.pregnancy_input)
        vbox.addWidget(self.glucose_label)
        vbox.addWidget(self.glucose_input)
        vbox.addWidget(self.blood_pressure_label)
        vbox.addWidget(self.blood_pressure_input)
        vbox.addWidget(self.skin_thickness_label)
        vbox.addWidget(self.skin_thickness_input)
        vbox.addWidget(self.insulin_label)
        vbox.addWidget(self.insulin_input)
        vbox.addWidget(self.bmi_label)
        vbox.addWidget(self.bmi_input)
        vbox.addWidget(self.diabetes_pedigree_function_label)
        vbox.addWidget(self.diabetes_pedigree_function_input)
        vbox.addWidget(self.age_label)
        vbox.addWidget(self.age_input)
        vbox.addWidget(self.predict_button)
        vbox.addWidget(self.output_label)

        # Set the layout of the main window
        self.setLayout(vbox)

        # Set the window properties
        self.setGeometry(300, 300, 350, 350)
        self.setWindowTitle("Diabetes Predictor: Random Forest")
        self.show()
    def predict(self):
        # Get the values from the input widgets
        pregnancy = self.pregnancy_input.text()
        glucose = self.glucose_input.text()
        blood_pressure = self.blood_pressure_input.text()
        skin_thickness = self.skin_thickness_input.text()
        insulin = self.insulin_input.text()
        bmi = self.bmi_input.text()
        diabetes_pedigree_function = self.diabetes_pedigree_function_input.text()
        age = self.age_input.text()

        # Make sure that all values are non-empty
        if not all([pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]):
            self.output_label.setText("Please enter values for all fields.")
            return

        # Make sure that all values are numbers
        try:
            pregnancy = float(pregnancy)
            glucose = float(glucose)
            blood_pressure = float(blood_pressure)
            skin_thickness = float(skin_thickness)
            insulin = float(insulin)
            bmi = float(bmi)
            diabetes_pedigree_function = float(diabetes_pedigree_function)
            age = float(age)
        except ValueError:
            self.output_label.setText("Please enter numbers for all fields.")
            return

        # Make a prediction using the classifier
        prediction = self.clf.predict([[pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])[0]
        if prediction == 0:
            self.output_label.setText("Person does not have diabetes.")
        else:
            self.output_label.setText("Person has diabetes.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DiabetesPredictor()
    sys.exit(app.exec_())


# In[ ]:




