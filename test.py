from src.classifier import LeftRightClassifier

MODEL_PATH = 'models/fundus_lr_classifier_resnet18.pth'
clf = LeftRightClassifier(model_path=MODEL_PATH)

print(clf.predict('samples/1.jpg'))
