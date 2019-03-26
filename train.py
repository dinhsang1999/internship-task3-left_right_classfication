from __future__ import division
import argparse
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import transforms, models

def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Left/Right fundus classification')
    parser.add_argument('--net_type', default='resnet', type=str, help='model: resnet, densenet')
    parser.add_argument('--image_height', default=224, type=int, help='image height to rescale')
    parser.add_argument('--image_width', default=336, type=int, help='image width to rescale')
    main()
