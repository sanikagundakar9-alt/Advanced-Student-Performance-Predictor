from flask import Flask, render_template, request, redirect
import numpy as np
import sqlite3
import joblib
from reportlab.pdfgen import canvas
import os

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Database setup
conn = sqlite3.connect('student.db')
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hours REAL,
    attendance REAL,
    previous_marks REAL,
    predicted_marks REAL
)
''')

conn.commit()
conn.close()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])
    previous_marks = float(request.form['previous_marks'])

    features = np.array([[hours, attendance, previous_marks]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    prediction = round(prediction, 2)

    # Smart suggestion
    if prediction < 60:
        suggestion = 'Increase study hours and attendance.'
    elif prediction < 80:
        suggestion = 'Good performance. Try to improve consistency.'
    else:
        suggestion = 'Excellent performance. Keep it up!'

    # Save to database
    conn = sqlite3.connect('student.db')
    cur = conn.cursor()

    cur.execute(
        '''
        INSERT INTO predictions
        (hours, attendance, previous_marks, predicted_marks)
        VALUES (?, ?, ?, ?)
        ''',
        (hours, attendance, previous_marks, prediction)
    )

    conn.commit()
    conn.close()

    return render_template(
        'index.html',
        prediction_text=f'Predicted Marks: {prediction}',
        suggestion=suggestion
    )


@app.route('/history')
def history():

    conn = sqlite3.connect('student.db')
    cur = conn.cursor()

    cur.execute('SELECT * FROM predictions')

    data = cur.fetchall()

    conn.close()

    return render_template('history.html', data=data)


@app.route('/report')
def report():

    if not os.path.exists('reports'):
        os.makedirs('reports')

    file_path = 'reports/student_report.pdf'

    c = canvas.Canvas(file_path)

    c.drawString(100, 800, 'Student Performance Prediction Report')
    c.drawString(100, 760, 'Generated using Flask + ML')

    c.save()

    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)