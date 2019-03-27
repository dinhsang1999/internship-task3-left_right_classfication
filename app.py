import os
import time
import tempfile

from flask_cors import CORS
from flask import Flask, request, redirect, url_for, json

from src.classifier import LeftRightClassifier

classifier = LeftRightClassifier()

app = Flask(__name__)
CORS(app)

# default route
@app.route('/')
def index():
    return "API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return app.response_class(
        response= """
        Wrong URL!
        <pre>{}</pre>""".format(e),
        status=404,
        mimetype='html/text'
    )

@app.errorhandler(500)
def server_error(e):
    return app.response_class(
        response= """
        An internal error occurred: <pre>{}</pre>
        See logs for full stacktrace.
        """.format(e),
        status=500,
        mimetype='html/text'
    )

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        response = app.response_class(
            response= 'File are not found',
            status=400,
            mimetype='html/text'
        )
        return response
    file = request.files['file']
    if file.filename == '':
        response = app.response_class(
            response= 'File are not found',
            status=400,
            mimetype='html/text'
        )
        return response

    # Store file to temp
    _, temp_filename = tempfile.mkstemp()
    file.save(temp_filename)

    response = app.response_class(
        response=json.dumps(classifier.predict(temp_filename)),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
