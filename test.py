import requests

# URL of your local Flask app
url = 'http://127.0.0.1:5000/predict'

# Sample text to test
text = "This is a sample text to test the model."

# Send a POST request to the Flask app with the sample text
response = requests.post(url, data={'text': text})

# Check if the request was successful
if response.status_code == 200:
    # Print the prediction
    print("Prediction:", response.json()['prediction'])
else:
    print("Error:", response.text)
