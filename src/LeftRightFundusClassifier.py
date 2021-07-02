import torch
from torch import nn
from torchvision import transforms
import torch.utils.model_zoo as model_zoo

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .model import LeftRightResnet18

model_urls = {
    "resnet18": "https://storage.googleapis.com/a2ds-models/lr_classifier/fundus_lr_classifier_resnet18-c18392c2.pth"
}

class LeftRightFundusClassifier():
    '''
    Left-Right Classifier for Fundus images

    '''
    def __init__(self, model_type='resnet18', model_path=None, using_gpu=True):
        '''
        Load trained model
        '''
        # Check device
        self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'
        # Create model
        self.model = LeftRightResnet18(True)
        # Convert to DataParallel and move to CPU/GPU
        self.model = nn.DataParallel(self.model).to(self.device)
        # Load trained model
        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        elif model_type in model_urls: # TODO: need refining
            self.model.load_state_dict(model_zoo.load_url(model_urls[model_type], map_location=self.device))

        # Switch model to evaluation mode
        self.model.eval()

        # Image processing
        self.height = 224
        self.width = self.height * 1.5
        self.transform = transforms.Compose([transforms.Resize((int(self.width), int(self.height))),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, image_input):
        '''
        Predict image in image_path is left or right
        '''
        # Read image
        if isinstance(image_input, str): # mean image path
            image = Image.open(image_input).convert('RGB')
        else: # means image PIL object # TODO: need to improve handling this, there can be case the input is neither of the case
            image = image_input

        # Transform image
        image = self.transform(image)
        image = image.view(1, *image.size()).to(self.device)
        # Result
        result = {'prob_left': 0, 'prob_right': 0}

        # Predict image
        with torch.no_grad():
            output = self.model(image)
            ps = output
            result['prob_left'] = float(ps[0][0].item())
            result['prob_right'] = float(ps[0][1].item())
        if abs(result['prob_left'] - result['prob_right']) < 0.9:
            result['label'] = 'undetermined'
        else:
            result['label'] = 'left' if result['prob_left'] > result['prob_right'] else 'right'
        return result
