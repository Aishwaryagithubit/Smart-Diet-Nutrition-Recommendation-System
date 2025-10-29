from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For Flask, prevents needing GUI
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ---- Load and clean dataset ----
df = pd.read_csv("nutrients.csv")

# Keep only needed columns
df = df[['Food', 'Calories', 'Protein', 'Carbs', 'Fat']]

# Convert to numeric, coerce errors to NaN, drop NaN
for col in ['Calories', 'Protein', 'Carbs', 'Fat']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

# ---- KNN model ----
features = df[['Calories', 'Protein', 'Carbs', 'Fat']]
knn_model = NearestNeighbors(n_neighbors=5)
knn_model.fit(features.values)

# ---- Recommendation function ----
def recommend_meals(target_cal):
    target_cal = min(max(target_cal, df['Calories'].min()), df['Calories'].max())
    idx = (df['Calories'] - target_cal).abs().argmin()
    _, indices = knn_model.kneighbors([features.values[idx]])
    recommended = df.iloc[indices[0]]
    return recommended

# ---- Function to create chart ----
def create_chart(recommended):
    plt.figure(figsize=(8,5))
    x = recommended['Food']
    plt.bar(x, recommended['Calories'], color='#0984e3', alpha=0.7, label='Calories')
    plt.bar(x, recommended['Protein'], color='#00b894', alpha=0.7, label='Protein')
    plt.bar(x, recommended['Carbs'], color='#fdcb6e', alpha=0.7, label='Carbs')
    plt.bar(x, recommended['Fat'], color='#d63031', alpha=0.7, label='Fat')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Amount (g / kcal)")
    plt.title("Nutrition Breakdown of Recommended Meals")
    plt.legend()
    plt.tight_layout()
    
    # Convert plot to PNG image for HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# ---- Flask routes ----
@app.route('/', methods=['GET', 'POST'])
def home():
    meals = None
    daily_cal = None
    chart_url = None
    if request.method == 'POST':
        weight = float(request.form['weight'])
        activity = request.form['activity']

        # Simple daily calorie estimation
        if activity == "sedentary":
            daily_cal = weight * 25
        elif activity == "moderate":
            daily_cal = weight * 30
        else:
            daily_cal = weight * 35

        target_cal = daily_cal / 3
        recommended = recommend_meals(target_cal)
        meals = recommended[['Food', 'Calories', 'Protein', 'Carbs', 'Fat']].values.tolist()

        # Create visualization chart
        chart_url = create_chart(recommended)

    return render_template('index.html', daily=daily_cal, meals=meals, chart_url=chart_url)

if __name__ == "__main__":
    app.run(debug=True)
