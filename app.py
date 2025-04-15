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

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Next.js

# Load environment variables
load_dotenv()

# Load model and tokenizer
model = tf.keras.models.load_model('problem_recommender.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# From your notebook
max_len = 963

def convert_verdict(verdict):
    return 1 if verdict == "OK" else 0

def dsconvert(tags):
    str = ""
    for i in tags:
        for j in i:
            if j == ' ':
                continue
            if j >= 'a' and j <= 'z':
                str += j
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
    Dataset = pd.DataFrame()
    encoded_handle = quote(handle)
    submission_url = f'https://codeforces.com/api/user.status?handle={encoded_handle}&from=1&count=10000'
    submission_request = requests.get(submission_url)

    if submission_request.status_code == 200:
        try:
            submission_data = submission_request.json()
            if submission_data.get('status') == 'OK':
                user_data = pd.DataFrame(submission_data.get('result'))
                user_data['handle'] = handle
                time.sleep(2)
                rating_url = f'https://codeforces.com/api/user.rating?handle={encoded_handle}'
                rating_request = requests.get(rating_url)
                if rating_request.status_code == 200:
                    rating_data = rating_request.json()
                    if rating_data.get('status') == 'OK':
                        rating = pd.DataFrame(rating_data.get('result'))
                        user_data['userRating'] = None
                        for index, row in user_data.iterrows():
                            creation_time = row['creationTimeSeconds']
                            filtered_ratings = rating[rating['ratingUpdateTimeSeconds'] < creation_time]
                            if not filtered_ratings.empty:
                                user_data.at[index, 'userRating'] = filtered_ratings.iloc[-1]['newRating']
                        Dataset = pd.concat([Dataset, user_data], ignore_index=True)
        except Exception as e:
            return None, str(e)
    else:
        return None, "Failed to fetch submission data"

    # Preprocess data
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

    # Process tags
    Dataset['tags'] = Dataset['tags'].apply(dsconvert)
    Dataset['tags'] = Dataset['tags'].str.split()
    Dataset = Dataset.explode('tags').dropna(subset=['tags']).reset_index(drop=True)
    Dataset['problemRating'] = Dataset['problemRating'].astype(int)
    Dataset['userRating'] = Dataset['userRating'].astype(int)
    Dataset['tags'] = Dataset['tags'].astype(str) + Dataset['problemRating'].astype(str)

    # Apply threshold
    Dataset = user_threshold(Dataset).reset_index(drop=True)

    return Dataset, None

def generate_recommendations(Dataset):
    user_result = Dataset.groupby('userID')['tags'].apply(lambda x: ' '.join(x)).reset_index()
    user_result['tags'] = user_result['tags'].apply(lambda x: ' '.join(x.split()[::-1]))
    text = user_result['tags'].astype(str).str.cat(sep=' ')
    recommendations = []
    for _ in range(10):  # Generate 10 recommendations
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len+1, padding='pre')
        pos = np.argmax(model.predict(padded_token_text, verbose=0))
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text += " " + word
                recommendations.append(word)
                break
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    handle = data.get('handle')
    if not handle:
        return jsonify({'error': 'Handle is required'}), 400

    Dataset, error = fetch_user_data(handle)
    if error:
        return jsonify({'error': error}), 500

    recommendations = generate_recommendations(Dataset)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)