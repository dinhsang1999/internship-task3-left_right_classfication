from pprint import pprint
from src.LeftRightFundusClassifier import LeftRightFundusClassifier

MODEL_PATH = 'models/fundus_lr_classifier_resnet18.pth'
clf = LeftRightFundusClassifier(model_path=MODEL_PATH)

pprint(clf.predict('samples/1.jpg'))