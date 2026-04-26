from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# paths to model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

# load everything once when server starts
print("Loading model and data...")
model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
courses_map = joblib.load(os.path.join(MODELS_DIR, 'courses_map.pkl'))
job_postings = pd.read_csv(os.path.join(MODELS_DIR, 'job_postings.csv'))
job_skills = pd.read_csv(os.path.join(MODELS_DIR, 'job_skills.csv'))
print("Everything loaded successfully!")


@app.route('/')
def home():
    return jsonify({'message': 'Career Recommender API is running!'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get user interests from frontend
        data = request.get_json()
        user_interests = data.get('interests', {})

        # build input vector with all features defaulting to 0
        input_row = {col: 0 for col in feature_cols}
        for key, val in user_interests.items():
            if key in input_row:
                input_row[key] = val

        # convert to numpy array
        input_array = pd.DataFrame([input_row])[feature_cols].values

        # get top 3 predictions with confidence scores
        probs = model.predict_proba(input_array)[0]
        top3_indices = probs.argsort()[-3:][::-1]
        top3_careers = le.inverse_transform(top3_indices)
        top3_probs = probs[top3_indices]

        # build predictions list
        predictions = []
        for career, prob in zip(top3_careers, top3_probs):
            predictions.append({
                'career': career,
                'confidence': round(float(prob) * 100, 2),
                'course': courses_map.get(career, 'N/A')
            })

        # job postings enrichment
        top_career = top3_careers[0]
        keywords = [k.strip().lower() for k in top_career.split(',')]
        matched_jobs = job_postings[
            job_postings['job_title'].str.lower().apply(
                lambda t: any(k in t for k in keywords)
            )
        ][['job_title', 'company', 'job_location', 'job_type']].head(5)

        # skills enrichment
        matched_skills = job_skills[
            job_skills['Title'].str.lower().str.contains(keywords[0], na=False)
        ][['Title', 'Minimum Qualifications']].head(3)

        return jsonify({
            'success': True,
            'predictions': predictions,
            'matched_jobs': matched_jobs.to_dict(orient='records'),
            'matched_skills': matched_skills.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({'features': feature_cols})


if __name__ == '__main__':
    app.run(debug=True, port=5000)