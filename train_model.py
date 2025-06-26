import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os

# --- Configuration ---
# IMPORTANT: Adjust this to the exact path and filename of your downloaded CSV
DATA_PATH = 'data/dataset.csv' # This should be correct for your setup
MODEL_DIR = 'model/'
NAIVE_BAYES_MODEL_PATH = os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
SYMPTOM_BINARIZER_PATH = os.path.join(MODEL_DIR, 'symptom_binarizer.pkl')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_models():
    """
    Loads disease-symptom data, preprocesses it, trains Naive Bayes and Random Forest
    models, and saves them along with the symptom binarizer.
    """
    print(f"Loading dataset from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}. Please ensure your CSV file is in the 'data/' directory with the correct name.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- Data Preprocessing ---
    # This should be correct for your dataset, based on previous debugging
    PROGNOSIS_COLUMN_NAME = 'Disease'

    if PROGNOSIS_COLUMN_NAME not in df.columns:
        print(f"Error: '{PROGNOSIS_COLUMN_NAME}' column not found in the dataset. Available columns: {df.columns.tolist()}")
        return

    symptom_cols = [col for col in df.columns if col != PROGNOSIS_COLUMN_NAME]

    all_symptoms_lists = []
    unique_symptoms_overall = set() # This set will still be useful for inspection/debugging

    for index, row in df.iterrows():
        symptoms_for_row = []
        for col in symptom_cols:
            symptom_val = str(row[col]).strip()
            if symptom_val and symptom_val.lower() != 'nan' and symptom_val.lower() != '':
                normalized_symptom = symptom_val.replace('_', ' ').lower()
                symptoms_for_row.append(normalized_symptom)
                unique_symptoms_overall.add(normalized_symptom)

        all_symptoms_lists.append(symptoms_for_row)

    # Convert unique symptoms set to a sorted list for consistent ordering (optional for mlb.fit_transform)
    # but useful for knowing the order of mlb.classes_ later.
    all_unique_symptoms_sorted = sorted(list(unique_symptoms_overall))

    print(f"Found {len(all_unique_symptoms_sorted)} unique symptoms in the dataset.")
    print(f"Example unique symptoms (first 10): {all_unique_symptoms_sorted[:10]}...")

    # --- CORRECTED: Initialize and fit/transform MultiLabelBinarizer ---
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(all_symptoms_lists) # This now correctly fits and transforms

    # It's good practice to ensure the classes of MLB are the sorted ones
    # Although fit_transform learns them, we might want to ensure a specific order if necessary later
    # For now, we rely on fit_transform to learn the classes from data itself.
    # If a fixed order is absolutely critical for some reason (e.g., external consistency),
    # you might need to manually set mlb.classes_ = all_unique_symptoms_sorted
    # after fit_transform, but usually fit_transform handles it fine.

    # Prepare target variable (diseases)
    y = df[PROGNOSIS_COLUMN_NAME].values

    # --- Train-Test Split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print("Warning: Cannot use stratification as there are too few samples in some classes. Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Number of diseases: {len(set(y))}")
    print(f"Example diseases (first 5): {list(set(y))[:5]}...")

    # --- Train Naive Bayes Model ---
    print("Training Naive Bayes Model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_accuracy = nb_model.score(X_test, y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

    # --- Train Random Forest Model ---
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    # --- Save Models and Binarizer ---
    print("Saving trained models and symptom binarizer...")
    joblib.dump(nb_model, NAIVE_BAYES_MODEL_PATH)
    joblib.dump(rf_model, RANDOM_FOREST_MODEL_PATH)
    joblib.dump(mlb, SYMPTOM_BINARIZER_PATH)
    print("Models and symptom binarizer saved successfully to the 'model/' directory.")

if __name__ == "__main__":
    train_and_save_models()