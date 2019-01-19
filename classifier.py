import torch
from torch import nn
from torchvision import transforms

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model import LeftRightResnet18

class LeftRightClassifier():
    '''
    Left-Right Classifier for Fundus images

    '''
    def __init__(self, model_path='./model/fundus_lr_classifier_resnet18.pth', using_gpu=True):
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
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        # Switch model to evaluation mode
        self.model.eval()

        # Image processing
        self.height = 224
        self.width = self.height * 1.5
        self.transform = transforms.Compose([transforms.Resize((int(self.width), int(self.height))),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, image_path):
        '''
        Predict image in image_path is left or right
        '''
        # Read image
        image = Image.open(image_path).convert('RGB')

        # Transform image
        image = self.transform(image)
        image = image.view(1, *image.size()).to(self.device)
        # Result
        result = {'prob_left': 0, 'prob_right': 0, 'prediction': 'left', 'need_check': True}
        labels = ['left', 'right']

        # Predict image
        with torch.no_grad():
            output = self.model(image)
            ps = torch.exp(output)
            result['prob_left'] = float(ps[0][0].item())
            result['prob_right'] = float(ps[0][1].item())
            _, top_class = ps.topk(1, dim=1)
            result['prediction'] = labels[int(top_class[0])]
        result['need_check'] = abs(result['prob_left'] - result['prob_right']) <= 0.5
        return result
