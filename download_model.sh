FILENAME='fundus_lr_classifier_resnet18.pth'
wget "https://drive.google.com/uc?export=download&id=1NdWsTVjHxl5DJ3zQtEQmsNiBVhh2zYLL" -O $FILENAME

mkdir models
mv $FILENAME models
