from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and clean dataset

df = pd.read_csv("nutrients.csv")
df.columns = df.columns.str.strip()

# Clean numeric columns
numeric_cols = ['Calories', 'Protein', 'Carbs', 'Fat']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(0)

# Add a "Type" column based on Category
def classify_type(category):
    category = str(category).lower()
    if any(word in category for word in ['meat', 'fish', 'egg', 'chicken']):
        return 'Non-Veg'
    elif any(word in category for word in ['milk', 'dairy', 'cheese']):
        return 'Veg'
    else:
        return 'Vegan'

df["Type"] = df["Category"].apply(classify_type)
df = df[['Food', 'Calories', 'Protein', 'Carbs', 'Fat', 'Type']]

# Meal recommendation

def recommend_meals(calories_needed, diet_type):
    filtered = df[df['Type'].str.lower() == diet_type.lower()]

    if filtered.empty:
        return []

    cal_values = filtered[['Calories']].values
    similarity = cosine_similarity(cal_values, [[calories_needed]])
    indices = similarity.argsort(axis=0)[-5:][::-1]  # top 5

    # Safe indexing
    if len(filtered) == 0 or len(indices) == 0 or indices[0][0] >= len(filtered):
        return []

    recommended = filtered.iloc[indices[:, 0]]
    return recommended.to_dict(orient='records')

# Nutrition chart

def create_chart(meals):
    if not meals:
        return None

    foods = [meal['Food'] for meal in meals]
    calories = [meal['Calories'] for meal in meals]
    protein = [meal['Protein'] for meal in meals]
    carbs = [meal['Carbs'] for meal in meals]
    fat = [meal['Fat'] for meal in meals]

    plt.figure(figsize=(8,5))
    bar_width = 0.2
    indices = range(len(foods))

    plt.bar([i - 1.5*bar_width for i in indices], calories, width=bar_width, label='Calories', color='#0984e3')
    plt.bar([i - 0.5*bar_width for i in indices], protein, width=bar_width, label='Protein', color='#00b894')
    plt.bar([i + 0.5*bar_width for i in indices], carbs, width=bar_width, label='Carbs', color='#fdcb6e')
    plt.bar([i + 1.5*bar_width for i in indices], fat, width=bar_width, label='Fat', color='#d63031')

    plt.xticks(indices, foods, rotation=45, ha='right')
    plt.ylabel('Amount (g / kcal)')
    plt.title('Nutrition Breakdown of Recommended Meals')
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return chart_url

# Home route

@app.route('/', methods=['GET', 'POST'])
def home():
    meals = None
    chart_url = None
    explanation = None

    if request.method == 'POST':
        try:
            weight = float(request.form['weight'])
            activity = request.form['activity']
            goal = request.form['goal']
            diet_type = request.form['diet']

            # Estimate daily calories
            if activity == "sedentary":
                daily_cal = weight * 22
            elif activity == "moderate":
                daily_cal = weight * 28
            else:
                daily_cal = weight * 33

            # Adjust based on goal
            if goal == "loss":
                daily_cal -= 300
            elif goal == "gain":
                daily_cal += 300

            # Generate meal plan (30% of daily calories per meal)
            part = 0.3
            meals = recommend_meals(daily_cal * part, diet_type)

            # Explanation
            explanation = "This meal plan balances calories, protein, carbs, and fats according to your selected goal and diet."

            # Generate chart
            chart_url = create_chart(meals)

        except Exception as e:
            return f"<h3>Error: {e}</h3>"

    return render_template('index.html', meals=meals, daily=daily_cal if meals else None,
                           diet=diet_type if meals else None, explanation=explanation, chart_url=chart_url)

# Generate PDF route

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    meals = request.form.getlist('meals')
    if not meals:
        return "No meals selected."

    doc_name = "recommended_meals.pdf"
    doc = SimpleDocTemplate(doc_name, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Recommended Meals Report", styles["Title"]), Spacer(1, 12)]

    data = [["Food", "Calories", "Protein", "Carbs", "Fat"]]
    for meal in meals:
        meal_data = meal.split(',')
        data.append(meal_data)

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0), colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    story.append(table)
    doc.build(story)

    return send_file(doc_name, as_attachment=True)

# Run Flask

if __name__ == '__main__':
    app.run(debug=True)


