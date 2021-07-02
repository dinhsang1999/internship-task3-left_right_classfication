from pprint import pprint
from src.classifier import LeftRightClassifier
import streamlit as st
import numpy as np
from PIL import Image

# Load model
MODEL_PATH = 'models/fundus_lr_classifier_resnet18.pth'
clf = LeftRightClassifier(model_path=MODEL_PATH)

# Render
st.sidebar.write('#### Upload an image')
uploaded_file = st.sidebar.file_uploader('', type=['png', 'jpg', 'jpeg'], 
										accept_multiple_files=False)

# Main component
st.write('# Fundus eye side reader')

if uploaded_file is None:
	# Default image
	image = np.ones((360, 640))
	st.image(image)
else:
	image = Image.open(uploaded_file)
	st.image(image)
	pred = clf.predict(image)
	st.write('Eyeside: ', pred['label'])
	st.write('P(left): ', round(pred['prob_left']*100, 2))
	st.write('P(right): ', round(pred['prob_right']*100, 2))

#pprint(clf.predict('samples/1.jpg'))
