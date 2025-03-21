# import libraries
from flask import Flask, request, jsonify, render_template_string
import pickle
from score import score
import warnings
import os

# ignore warnings
warnings.filterwarnings("ignore")

# create flask app
app = Flask(__name__)

# load model
# model = pickle.load(open('/home/utpalraj/coursework/AML/AppliedMachineLearning/Assignment3/model/best_model.pkl', "rb"))
filename = os.path.join(os.path.dirname(__file__), "model", "best_model.pkl")
model = pickle.load(open(filename, "rb"))
# html template
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Classifier</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            flex-direction: column;
            background-color: #f0f0f0;
        }
        h1 {
            text-align: center;
            margin-top: 20px;
        }
        form {
            text-align: center;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        label {
            display: block;
            margin-bottom: 10px;
        }
        input[type="text"] {
            width: 100%;
            height: 40px;
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        input[type="submit"] {
            width: 100%;
            height: 40px;
            background-color: #4CAF50;
            color: #fff;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #3e8e41;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <h1>Spam Classifier</h1>
    <form action="/" method="post">
        <label for="text">Enter Text:</label>
        <input type="text" id="text" name="text">
        <input type="submit" value="Classify">
    </form>
    {% if prediction %}
        <div class="result">
            <h2>Result:</h2>
            <p>Prediction: {{ prediction }}</p>
            <p>Propensity: {{ probability }}</p>
        </div>
    {% endif %}
</body>
</html>
"""


# home route
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handle requests to the home route. 

    For GET requests, renders the HTML page for the spam classifier.
    For POST requests, processes the submitted text, scores it using 
    the model, and returns the prediction and propensity as a JSON response.

    Returns:
        JSON: A JSON response for POST requests containing the prediction 
              and propensity.
        str: An HTML string for GET requests to render the spam classifier page.
    """

    if request.method == 'POST':
        text = request.form['text']

        prediction, probability = score(text, model, 0.50)

        response = {
            "prediction": prediction,
            "propensity": probability
        }
        
        return jsonify(response)
    
    else:
        return render_template_string(html)

# run app
if __name__ == '__main__':
    app.run(debug=True)
