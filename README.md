# Smart Diet & Nutrition Recommendation System - Personalized Meal Planner

Smart Diet & Nutrition is a simple yet intelligent web application that generates personalized meal plans based on a user’s weight, activity level, diet preference, and fitness goal.
The app uses dynamic HTML templates (Jinja2), calculates nutritional values for the day, and provides:
✔️ Fullday meal plan
✔️ Calorie breakdown
✔️ Auto generated nutrition chart
✔️ PDF export of the meal plan
✔️ Explanation of why the plan suits the user

🚀 Features
🔹 User Inputs

Weight (kg)

Activity Level: Sedentary / Moderate / Active

Diet Preference: Veg / Non-Veg / Vegan

Goal: Weight Loss or Muscle Gain

🔹 Outputs

Full day meal plan in a tabular format

Calories, Protein, Carbs, Fat for each meal

Auto generated nutrition visualization chart 

PDF download option for meal plan

Explanation box showing why the plan fits the user’s health goal

🛠️ Tech Stack

Frontend: HTML5, CSS3, FontAwesome Icons

Backend: Python 

Templating: Jinja2

Charts: Matplotlib

PDF Generation: ReportLab or FPDF

Styling: Clean, modern UI with Poppins-like design

📂 Project Structure
project-folder/
│── app.py
│── templates/
│     └── index.html
│── static/
│     └── (CSS / images if needed)
│── meal_data.py
│── requirements.txt
│── README.md

