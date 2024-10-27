import pickle
from flask import Flask, request, render_template
import pandas as pd


app = Flask(__name__)

# Load the saved model and preprocessors
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

def preprocess_input(data):
    """Preprocess the input data using saved preprocessors"""
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Categorical features
    categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                          'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                          'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                          'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    
    # Numerical features
    numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
                         'Tutoring_Sessions', 'Physical_Activity']
    
    # Encode categorical features
    for feature in categorical_features:
        le = label_encoders[feature]
        df[feature] = le.transform(df[feature])
    
    # Scale numerical features
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    return df

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Collect form data
        input_data = {
            'Hours_Studied': float(request.form['Hours_Studied']),
            'Attendance': float(request.form['Attendance']),
            'Parental_Involvement': request.form['Parental_Involvement'],
            'Access_to_Resources': request.form['Access_to_Resources'],
            'Extracurricular_Activities': request.form['Extracurricular_Activities'],
            'Sleep_Hours': float(request.form['Sleep_Hours']),
            'Previous_Scores': float(request.form['Previous_Scores']),
            'Motivation_Level': request.form['Motivation_Level'],
            'Internet_Access': request.form['Internet_Access'],
            'Tutoring_Sessions': float(request.form['Tutoring_Sessions']),
            'Family_Income': request.form['Family_Income'],
            'Teacher_Quality': request.form['Teacher_Quality'],
            'School_Type': request.form['School_Type'],
            'Peer_Influence': request.form['Peer_Influence'],
            'Physical_Activity': float(request.form['Physical_Activity']),
            'Learning_Disabilities': request.form['Learning_Disabilities'],
            'Parental_Education_Level': request.form['Parental_Education_Level'],
            'Distance_from_Home': request.form['Distance_from_Home'],
            'Gender': request.form['Gender']
        }
        
        # Preprocess the input
        processed_input = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction = round(prediction, 2)
    
    return render_template('index.html', prediction=prediction)

# Save the best model (add this to your ML code)
def save_model(best_model, scaler, label_encoders_dict):
    """
    Save the model and preprocessors
    """
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
        
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
        
    with open('label_encoders.pkl', 'wb') as file:
        pickle.dump(label_encoders_dict, file)

if __name__ == '__main__':
    app.run(debug=True)