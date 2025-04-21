from flask import Flask, render_template, request

app = Flask(__name__)

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
dataset = pd.read_csv('./heart.csv')

# Separate the features (predictor variables) and the target variable
X = dataset.drop('output', axis=1)  # Adjust 'target_variable_name' to your column name
y = dataset['output']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % accuracy)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input data from the form
        age = int(request.form['age'])
        sex = int(request.form['gender'])
        cp = int(request.form['chest-pain'])
        trestbps = int(request.form['rbp'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Making predictions using the trained model
        new_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = model.predict(new_data)

        if prediction == 1:
            result = 'The patient is likely to have heart disease.'
        else:
            result = 'The patient is unlikely to have heart disease.'

        return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
