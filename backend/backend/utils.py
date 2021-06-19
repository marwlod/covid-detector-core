import os
import tempfile

import cv2
import torch
import torchvision
from torchvision import transforms

MODEL_NAME = "resnet50_V2.pt"


def get_image_from_request(request):
    image = request.FILES['file']
    temp, temp_name = tempfile.mkstemp()
    for chunk in image.chunks():
        os.write(temp, chunk)
    rgb_img = cv2.imread(temp_name, 1)[:, :, ::-1]
    os.close(temp)
    return rgb_img


def load_model():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=3)
    model.load_state_dict(torch.load("../" + MODEL_NAME))
    return model


def image_to_tensor(image, resize=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if resize:
        test_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std)])
    else:
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std)])
    image_tensor = test_transforms(image)
    return image_tensor.unsqueeze(0)