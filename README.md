# Student AI Project

A Flask web app that predicts a student's final marks using a trained machine learning model.

## Features

- Predicts final marks from:
  - Attendance
  - Internal marks
  - Previous GPA
  - Daily study hours
- Displays risk level based on predicted marks:
  - High risk: below 50
  - Medium risk: 50 to below 70
  - Low risk: 70 and above
- Provides simple recommendations based on attendance and study hours.

## Tech Stack

- Python
- Flask
- pandas
- scikit-learn

## Project Structure

- `app.py` - Flask app, routes, prediction, and recommendation logic
- `model.py` - Model training and saving logic
- `dataset.csv` - Training data
- `model.pkl` - Saved trained model
- `templates/index.html` - Web UI template
- `static/style.css` - Styling

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install flask pandas scikit-learn
```

## Run the App

```bash
python app.py
```

## Quick Run (Next Time)

From `C:\Users\Manoj\Desktop\student-ai-project` in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python app.py
```

Then open:

- http://127.0.0.1:5000/

To stop the app, press `Ctrl + C` in the same terminal.

If dependency versions change and you need to refresh once:

```powershell
pip install flask pandas scikit-learn
```

Then open your browser at:

- http://127.0.0.1:5000/

## How It Works

- On startup, the app checks for `model.pkl`.
- If the model file does not exist, it trains a new RandomForestRegressor model using `dataset.csv` and saves it.
- The `/predict` route accepts form input, runs prediction, computes risk level, and shows recommendations.

## Input Ranges

- Attendance: 0 to 100
- Internal Marks: 0 to 100
- GPA: 0 to 10
- Study Hours: 0 and above

## Notes

- Flask runs in debug mode in `app.py` (`app.run(debug=True)`).
- For production use, disable debug mode and run with a production WSGI server.
