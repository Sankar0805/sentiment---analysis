from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


def train_and_save_model():
	df = pd.read_csv(DATASET_PATH)

	feature_columns = [
		"attendance",
		"internal_marks",
		"previous_gpa",
		"study_hours",
	]
	target_column = "final_marks"

	missing = [col for col in feature_columns + [target_column] if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in dataset.csv: {missing}")

	X = df[feature_columns]
	y = df[target_column]

	model = RandomForestRegressor(n_estimators=200, random_state=42)
	model.fit(X, y)

	with MODEL_PATH.open("wb") as model_file:
		pickle.dump(model, model_file)

	return model
