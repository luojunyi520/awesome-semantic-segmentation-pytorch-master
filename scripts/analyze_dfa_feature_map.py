import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from core.models import get_model
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='dfanet_citys',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='citys', choices=['pascal_voc, pascal_aug, ade20k, citys'],
                    help='dataset name (default: pascal_voc)')

args = parser.parse_args()

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

model = get_model(args.model, pretrained=True).to('cuda')
print(model)

# load image
img = Image.open("./per00001.jpg")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    print(im.shape)
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        # plt.imshow(im[:, :, i], cmap='gray')
        plt.imshow(im[:, :, i])
    plt.show()

