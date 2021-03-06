# Left Right Fundus classification

## Requirements
- Python >=3.5

```bash
conda create -n eyeside python=3.7
conda activate eyeside
```

## Setup environment
Run this script to create a virtual environment and install dependency libraries

```bash
pip install -r requirements.txt
```

Download model
```bash
bash download_model.sh
```

## Predict image
Simply import the module and create a classifier object:
```python
from LeftRightFundusClassifier import LeftRightFundusClassifier

MODEL_PATH = 'models/fundus_lr_classifier_resnet18.pth'
clf = LeftRightFundusClassifier(model_path=MODEL_PATH)

print(clf.predict('../samples/1.jpg'))
```

The `result` is a dictionary with the structure:
```javascript
{
  'prob_left': 0.007223138585686684, // Probability of image is of left eye
  'prob_right': 0.9927768707275391, // Probability of image is of right eye
  'label': 'right' // Label from classifier: 'left' or 'right' or 'undetermined'
}
```

## Streamlit app
```bash
streamlit run streamlit_app.py
```

## Flask API
Run this command
```bash
gunicorn -b 0.0.0.0:8000 app
```
Open browser and call API on port 8000
- METHOD: GET
- URL: http://localhost:8000/fundus/left-right/predict
- PARAMS:
  - `id`: image name in images folder

## Docker Installation

### Build and run

#### non-GPU docker
1. Install docker, run docker
2. Build docker
```bash
./docker-build.sh <version>
```
3. Run docker on port 5001. You can pass any port you want to run
```bash
./docker-run.sh 5001
```
4. Open host browser on port 5001

#### GPU docker
1. Install docker, run docker
2. Config nvidia-docker https://github.com/NVIDIA/nvidia-docker
3. Build docker
```bash
./docker-gpu-build.sh <version>
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

### Push Docker images

1. Install Cloud SDK https://cloud.google.com/sdk/docs/
2. Config docker with Google Cloud https://cloud.google.com/container-registry/docs/advanced-authentication
3. Push docker
```bash
./docker-push.sh
./docker-gpu-push.sh
```
