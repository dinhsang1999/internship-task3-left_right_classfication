# Left Right Fundus classification

## Setup environment
Run this script to create a virtual environment and install dependency libraries

```bash
pip3 install -r requirements.txt
```
or
```bash
pip install -r requirements.txt
```

## Predict image
Simply import the module and create a classifier object:
```python
from classifier import LeftRightClassifier

clf = LeftRightClassifier()
```
Predict image at `image_path` you want:
```python
result = clf.predict('path/to/your/image')
```
The `result` is a dictionary with the structure:
```javascript
{
  'prob_left': 0.007223138585686684, // Probability of image is of left eye
  'prob_right': 0.9927768707275391, // Probability of image is of right eye
}
```
