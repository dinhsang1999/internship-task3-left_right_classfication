import os
import time
import tempfile

from flask_cors import CORS
from flask import Flask, request, redirect, url_for, json

from src.LeftRightFundusClassifier import LeftRightFundusClassifier

MODEL_PATH = 'models/fundus_lr_classifier_resnet18.pth'
clf = LeftRightFundusClassifier(model_path=MODEL_PATH)

IMAGE_FOLDER = os.path.abspath('images')

application = Flask(__name__)
CORS(application)

# default route
@application.route('/')
def index():
    return "API"

# HTTP Errors handlers
@application.errorhandler(404)
def url_error(e):
    return application.response_class(
        response= """
        Wrong URL!
        <pre>{}</pre>""".format(e),
        status=404,
        mimetype='html/text'
    )

@application.errorhandler(500)
def server_error(e):
    return application.response_class(
        response= """
        An internal error occurred: <pre>{}</pre>
        See logs for full stacktrace.
        """.format(e),
        status=500,
        mimetype='html/text'
    )

@application.route('/fundus/left-right/predict', methods=['GET'])
def predict():
    imageId = user = request.args.get('id')
    if imageId is None:
        return application.response_class(
            response= 'Missing id in request',
            status=422,
            mimetype='html/text'
        )
    image_path = os.path.join(IMAGE_FOLDER, imageId)
    if not os.path.exists(image_path):
        return application.response_class(
            response= 'Image not found',
            status=400,
            mimetype='html/text'
        )

    response = application.response_class(
        response=json.dumps(clf.predict(image_path)),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)
