from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("C:/Users/Hiruni/OneDrive/Desktop/thyroid_app/models/thyroid_model.pkl", "rb"))
columns = pickle.load(open("C:/Users/Hiruni/OneDrive/Desktop/thyroid_app/models/model_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html", prediction_text="?")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        data = {
            'Age': int(request.form['Age']),
            'Gender': request.form['Gender'],
            'Country': request.form['Country'],
            'Ethnicity': request.form['Ethnicity'],
            'Family_History': request.form['Family_History'],
            'Radiation_Exposure': request.form['Radiation_Exposure'],
            'Iodine_Deficiency': request.form['Iodine_Deficiency'],
            'Smoking': request.form['Smoking'],
            'Obesity': request.form['Obesity'],
            'Diabetes': request.form['Diabetes'],
            'TSH_Level': float(request.form['TSH_Level']),
            'T3_Level': float(request.form['T3_Level']),
            'T4_Level': float(request.form['T4_Level']),
            'Nodule_Size': float(request.form['Nodule_Size']),
        }

        # Binary encoding
        binary_map = {"No": 0, "Yes": 1, "Male": 0, "Female": 1}
        for col in ['Gender', 'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']:
            data[col] = binary_map[data[col]]

        df = pd.DataFrame([data])
        df = pd.get_dummies(df, columns=['Country', 'Ethnicity'])
        df = df.reindex(columns=columns, fill_value=0)

        prediction = model.predict(df)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        return render_template("index.html",
                               prediction_text=f"{result}")

if __name__ == "__main__":
    app.run(debug=True)
