from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
import time
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": "https://codeforces-recommender.vercel.app"}})

logger.info("Starting application...")

load_dotenv()

try:
    logger.info("Loading model...")
    model = tf.keras.models.load_model('problem_recommender.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

try:
    logger.info("Loading tokenizer...")
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {str(e)}")
    raise

max_len = 963

def convert_verdict(verdict):
    return 1 if verdict == "OK" else 0

def dsconvert(tags):
    str = ""
    for i in tags:
        for j in i:
            if j == ' ': continue
            if j >= 'a' and j <= 'z': str += j
        str += " "
    return str

def user_threshold(group):
    tag_counts = {}
    result = []
    for index, row in group.iterrows():
        tag = row['tags']
        if tag not in tag_counts:
            tag_counts[tag] = 0
        if tag_counts[tag] < 2:
            result.append(row)
            tag_counts[tag] += 1
    return pd.DataFrame(result)

def fetch_user_data(handle):
    logger.info(f"Fetching data for handle: {handle}")
    Dataset = pd.DataFrame()
    encoded_handle = quote(handle)
    submission_url = f'https://codeforces.com/api/user.status?handle={encoded_handle}&from=1&count=10000'
    for attempt in range(3):
        try:
            submission_request = requests.get(submission_url, timeout=10)
            if submission_request.status_code == 200:
                break
            logger.warning(f"Attempt {attempt + 1} failed for submission data")
            time.sleep(2)
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            time.sleep(2)
    else:
        logger.error("Failed to fetch submission data after retries")
        return None, "Failed to fetch submission data"

    try:
        submission_data = submission_request.json()
        if submission_data.get('status') == 'OK' and submission_data.get('result'):
            user_data = pd.DataFrame(submission_data.get('result'))
            user_data['handle'] = handle
            time.sleep(2)
            rating_url = f'https://codeforces.com/api/user.rating?handle={encoded_handle}'
            rating_request = requests.get(rating_url, timeout=10)
            if rating_request.status_code == 200:
                rating_data = rating_request.json()
                if rating_data.get('status') == 'OK' and rating_data.get('result'):
                    rating = pd.DataFrame(rating_data.get('result'))
                    user_data['userRating'] = None
                    for index, row in user_data.iterrows():
                        creation_time = row['creationTimeSeconds']
                        filtered_ratings = rating[rating['ratingUpdateTimeSeconds'] < creation_time]
                        if not filtered_ratings.empty:
                            user_data.at[index, 'userRating'] = filtered_ratings.iloc[-1]['newRating']
                    Dataset = pd.concat([Dataset, user_data], ignore_index=True)
                else:
                    logger.warning(f"No rating data for {handle}")
                    return None, "No rating data found"
            else:
                logger.error("Failed to fetch rating data")
                return None, "Failed to fetch rating data"
        else:
            logger.warning(f"No submissions for {handle}")
            return None, f"No submission data found for handle: {handle}"
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        return None, f"Error processing data: {str(e)}"

    try:
        Dataset[['contestId', 'problemsetName', 'index', 'name', 'type', 'points', 'problemRating', 'tags']] = pd.DataFrame(
            Dataset['problem'].apply(lambda x: [x.get('contestId'), x.get('problemsetName'), x.get('index'),
                                               x.get('name'), x.get('type'), x.get('points'),
                                               x.get('rating'), x.get('tags')]).tolist())
        Dataset = Dataset.fillna(0)
        Dataset['contestId'] = Dataset['contestId'].astype(int)
        Dataset['Problem'] = Dataset['contestId'].astype(str) + Dataset['index'].astype(str) + ' ' + Dataset['name']
        Dataset['attempts'] = Dataset.groupby(['handle', 'Problem'])['handle'].transform('size')
        Dataset = Dataset.drop_duplicates(subset=['handle', 'Problem'], keep='first').reset_index(drop=True)
        Dataset['userID'] = 1
        columns = ['Problem', 'problemRating', 'verdict', 'userRating', 'attempts', 'userID', 'tags']
        Dataset['verdict'] = Dataset['verdict'].apply(convert_verdict)
        Dataset = Dataset[columns]

        Dataset['tags'] = Dataset['tags'].apply(dsconvert)
        Dataset['tags'] = Dataset['tags'].str.split()
        Dataset = Dataset.explode('tags').dropna(subset=['tags']).reset_index(drop=True)
        Dataset['problemRating'] = Dataset['problemRating'].astype(int)
        Dataset['userRating'] = Dataset['userRating'].astype(int)
        Dataset['tags'] = Dataset['tags'].astype(str) + Dataset['problemRating'].astype(str)
        Dataset = user_threshold(Dataset).reset_index(drop=True)
        logger.info(f"Data processed for {handle}: {len(Dataset)} rows")
        return Dataset, None
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return None, f"Error preprocessing data: {str(e)}"

def generate_recommendations(Dataset):
    logger.info("Generating recommendations...")
    try:
        user_result = Dataset.groupby('userID')['tags'].apply(lambda x: ' '.join(x)).reset_index()
        user_result['tags'] = user_result['tags'].apply(lambda x: ' '.join(x.split()[::-1]))
        text = user_result['tags'].astype(str).str.cat(sep=' ')
        recommendations = []
        for _ in range(10):
            token_text = tokenizer.texts_to_sequences([text])[0]
            padded_token_text = pad_sequences([token_text], maxlen=max_len+1, padding='pre')
            pos = np.argmax(model.predict(padded_token_text, verbose=0))
            for word, index in tokenizer.word_index.items():
                if index == pos:
                    text += " " + word
                    recommendations.append(word)
                    break
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise Exception(f"Error generating recommendations: {str(e)}")

@app.route('/recommend', methods=['POST'])
def recommend():
    logger.info("Received /recommend request")
    try:
        data = request.get_json()
        handle = data.get('handle')
        if not handle:
            logger.warning("No handle provided")
            return jsonify({'error': 'Handle is required'}), 400
        Dataset, error = fetch_user_data(handle)
        if error:
            logger.warning(f"Fetch error: {error}")
            return jsonify({'error': error}), 400
        recommendations = generate_recommendations(Dataset)
        logger.info("Request successful")
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check")
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask on port {port}")
    app.run(host='0.0.0.0', port=port)