from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import os
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key for session management

# MongoDB Configuration
MONGO_URI = 'mongodb://localhost:27017/'  # Replace with your MongoDB URI
client = MongoClient(MONGO_URI)
db = client['user_database']  # Replace with your database name
users_collection = db['users']


# Paths
MODEL_PATH = 'best_modeel.pkl'  # Replace with the actual path to your trained model
DATASET_PATH = 'cleaneddata.csv'  # Replace with the actual path to your dataset
data = pd.read_csv(DATASET_PATH)


# Load the trained model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

# Load the dataset (if you need it for further operations)
if os.path.exists(DATASET_PATH):
    dataset = pd.read_csv(DATASET_PATH)
else:
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")


@app.route('/')
def home():
    # Redirect to the login page as the default page
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email=request.form['email']
        phonenumber=request.form['phonenumber']
        password = request.form['password']


        # Check if the username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists. Please choose another one.', 'error')
            return redirect(url_for('register'))

        # Insert the new user into the database
        users_collection.insert_one({'username': username,'email':email,'phonenumber':phonenumber,'password': password})
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Authenticate the user
        user = users_collection.find_one({'username': username, 'password': password})
        if user:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  # Redirect to the index page after login
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')


@app.route('/index')
def index():
    if 'username' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('index.html')



@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        inputs = {
            "gender": request.form['gender'],
            "battery_last": request.form['battery_last'],
            "charger_run": request.form['charger_run'],
            "phone_worry": request.form['phone_worry'],
            "bathroom_phone": request.form['bathroom_phone'],
            "social_phone": request.form['social_phone'],
            "phone_no_notification": request.form['phone_no_notification'],
            "sleep_phone": request.form['sleep_phone'],
            "class_time_check": request.form['class_time_check'],
            "awkward_rely": request.form['awkward_rely'],
            "tv_eating_phone": request.form['tv_eating_phone'],
            "panic_phone": request.form['panic_phone'],
            "phone_on_date": request.form['phone_on_date'],
            "game_usage": request.form['game_usage'],
            "can_live_without_phone": request.form['can_live_without_phone']
        }

        def map_yes_no(value):
            return 1 if value.lower() == 'yes' else 0

        def map_gender(value):
            if value == 'Male':
                return 0
            elif value == 'Female':
                return 1
            elif value == 'Other':
                return 2
            else:
                return -1
        behavioral_score = sum([
            map_yes_no(inputs['bathroom_phone']),
            map_yes_no(inputs['social_phone']),
            map_yes_no(inputs['phone_no_notification']),
            map_yes_no(inputs['sleep_phone']),
            map_yes_no(inputs['class_time_check'])
        ]) * 2
        dependency_score = sum([
            1 - map_yes_no(inputs['battery_last']),  # 'No' adds to the score
            map_yes_no(inputs['charger_run']),
            map_yes_no(inputs['phone_worry']),
            1 - map_yes_no(inputs['can_live_without_phone']),  # 'No' adds to the score
            map_yes_no(inputs['game_usage'])  # Assuming 'game_usage' is a numeric input
        ]) * 2

        social_psychological_score = sum([
            map_yes_no(inputs['awkward_rely']),
            map_yes_no(inputs['tv_eating_phone']),
            map_yes_no(inputs['panic_phone']),
            map_yes_no(inputs['phone_on_date'])
        ]) * 3


        # Total score
        total_score = behavioral_score + dependency_score + social_psychological_score

        # Map inputs for prediction
        mapped_inputs = [
            map_gender(inputs['gender']),
            map_yes_no(inputs['battery_last']),
            map_yes_no(inputs['charger_run']),
            map_yes_no(inputs['phone_worry']),
            map_yes_no(inputs['bathroom_phone']),
            map_yes_no(inputs['social_phone']),
            map_yes_no(inputs['phone_no_notification']),
            map_yes_no(inputs['sleep_phone']),
            map_yes_no(inputs['class_time_check']),
            map_yes_no(inputs['awkward_rely']),
            map_yes_no(inputs['tv_eating_phone']),
            map_yes_no(inputs['panic_phone']),
            map_yes_no(inputs['phone_on_date']),
            map_yes_no(inputs['game_usage']),
            map_yes_no(inputs['can_live_without_phone']),
        ]

        # Prediction
        prediction = model.predict([mapped_inputs])[0]
        addiction_status = "Addicted" if prediction == 2 and total_score > 15 else "Not Addicted"

        warnings = []
        recommendations = []

        if total_score >= 23:
            warnings.append("Severe phone addiction detected! Immediate action is recommended.")
        elif total_score >= 15 and total_score < 23:
            warnings.append("Moderate phone addiction detected. Monitor your usage patterns closely.")
        elif total_score >= 7 and total_score < 15:
            warnings.append("Mild phone addiction detected. Adjust your habits to avoid escalation.")

        # Behavioral insights
        behavioral_analysis = []
        if map_yes_no(inputs['social_phone']) == 1:
            behavioral_analysis.append("Frequent phone use during social gatherings.")
        if map_yes_no(inputs['sleep_phone']) == 1:
            behavioral_analysis.append("Using your phone before sleep or after waking up may affect sleep quality.")
        if map_yes_no(inputs['tv_eating_phone']) == 1:
            behavioral_analysis.append("Using your phone while eating or watching TV may reduce mindfulness.")

        # Recommendations based on behaviors
        if map_yes_no(inputs['phone_no_notification']) == 1:
            recommendations.append("Consider turning off non-essential notifications to improve focus.")
        if map_yes_no(inputs['game_usage']) == 1:
            recommendations.append("Limit gaming sessions to avoid overuse.")
        if map_yes_no(inputs['bathroom_phone']) == 1:
            recommendations.append("Try avoiding taking your phone to the bathroom to reduce dependency.")

        return render_template(
            'results.html',
            addiction_status=addiction_status,
            behavioral_score=behavioral_score,
            dependency_score=dependency_score,
            social_psychological_score=social_psychological_score,
            phone_score=total_score,
            behavioral_analysis=behavioral_analysis,
            warnings=warnings,
            recommendations=recommendations
        )

    return render_template('predict.html')

import plotly.express as px
import plotly.graph_objects as go
@app.route('/visualize')
def visualize():
    # Gender Distribution
    gender_counts = data['Gender :'].value_counts()
    gender_fig = px.pie(names=gender_counts.index, values=gender_counts.values, title="Gender Distribution")
    
    # Phone Addiction
    addiction_counts = data['whether you are addicted to phone?'].value_counts()
    addiction_fig = px.bar(x=addiction_counts.index, y=addiction_counts.values, title="Phone Addiction Rate", text=addiction_counts.values)
    addiction_fig.update_traces(marker_color=['red', 'green'], textposition='outside')
    
    # Phone Usage Before Sleep
    sleep_usage_counts = data['Do you check your phone just before going to sleep/just after waking up?'].value_counts()
    sleep_usage_fig = px.bar(x=sleep_usage_counts.index, y=sleep_usage_counts.values, title="Phone Usage Before Sleep", text=sleep_usage_counts.values)
    sleep_usage_fig.update_traces(marker_color='blue', textposition='outside')
    
    # Game Time
    game_time_counts = data['For how long do you use your phone for playing games?'].value_counts()
    game_time_fig = px.pie(names=game_time_counts.index, values=game_time_counts.values, title="Game Time Distribution")
    
    # Dependency Analysis
    dependency_questions = ['Do you worry about losing your cell phone?', 'Do you take your phone to the bathroom?', 'Do you have a panic attack if you leave your phone elsewhere?']
    dependency_counts = {q: data[q].value_counts() for q in dependency_questions}
    dependency_fig = go.Figure()
    for q, counts in dependency_counts.items():
        dependency_fig.add_trace(go.Bar(name=q, x=counts.index, y=counts.values))
    dependency_fig.update_layout(title_text="Phone Dependency Metrics", barmode='group')
    
    # Convert plots to JSON
    graphs = {
        "gender_fig": gender_fig.to_html(full_html=False),
        "addiction_fig": addiction_fig.to_html(full_html=False),
        "sleep_usage_fig": sleep_usage_fig.to_html(full_html=False),
        "game_time_fig": game_time_fig.to_html(full_html=False),
        "dependency_fig": dependency_fig.to_html(full_html=False)
    }
    
    return render_template('visualization.html', graphs=graphs)

@app.route('/metrics')
def show_metrics():
    return render_template('metrics.html')



if __name__ == '__main__':
    app.run(debug=True)
