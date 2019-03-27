# Left Right Fundus classification

## Requirements
- Python >=3.5

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
  'label': 'right' // Label from classifier: 'left' or 'right' or 'undetermined'
}
```

## Flask API
Run this command
```bash
python app.py
```
Open browser and call API on port 5000
- METHOD: POST
- URL: http://localhost:5000/predict
- DATA:
```
{
  "file": "<Image file>"
}
```

## Docker Installation

### non-GPU docker
1. Install docker, run docker
2. Build docker
```bash
./docker-build.sh
```
3. Run docker on port 5001. You can pass any port you want to run
```bash
./docker-run.sh 5001
```
4. Open host browser on port 5001

### GPU docker
1. Install docker, run docker
2. Config nvidia-docker https://github.com/NVIDIA/nvidia-docker
3. Build docker
```bash
./docker-gpu-build.sh
```
4. Test nvidia docker. At this step, you will see your GPU appear on the console
```bash
docker run --runtime=nvidia --rm nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 nvidia-smi
```
5. Run docker on port 5002. You can pass any port you want to run
```bash
./docker-gpu-run.sh 5002
```
6. Open host browser on port 5002
