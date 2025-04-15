import pandas as pd
import requests
import time
from urllib.parse import quote

def fetch_user_data(user_handle):
    encoded_handle = quote(user_handle)
    submission_url = f"https://codeforces.com/api/user.status?handle={encoded_handle}&from=1&count=10000"
    submission_request = requests.get(submission_url)
    submission_data = submission_request.json()

    if submission_data.get("status") != "OK":
        raise Exception("Error fetching submission data")

    user_data = pd.DataFrame(submission_data["result"])
    user_data["handle"] = user_handle

    # Add rating info and preprocessing here
    # Return the processed dataframe ready for prediction
    return user_data  # Processed features used in model
