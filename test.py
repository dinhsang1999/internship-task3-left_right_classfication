from pprint import pprint
from src.classifier import LeftRightClassifier

MODEL_PATH = 'models/fundus_lr_classifier_resnet18.pth'
clf = LeftRightClassifier(model_path=MODEL_PATH)

pprint(clf.predict('samples/1.jpg'))
