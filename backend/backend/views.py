import io
import json
import os
import tempfile

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

MODEL_NAME = "resnet50_V2.pt"


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


def predict(model, image):
    model.eval()
    image_tensor = image_to_tensor(image)
    inp = Variable(image_tensor)
    output = model(inp)
    m = nn.Softmax(dim=1)
    return m(output)


@csrf_exempt
def classify(request):
    image = request.FILES['file']
    temp, temp_name = tempfile.mkstemp()
    for chunk in image.chunks():
        os.write(temp, chunk)

    model = load_model()
    rgb_img = cv2.imread(
        temp_name, 1)[:, :, ::-1]
    os.close(temp)
    image = Image.fromarray(rgb_img, 'RGB')
    prediction = predict(model, image)
    response = {
        'normal': prediction[0][0].tolist(),
        'viral': prediction[0][1].tolist(),
        'covid': prediction[0][2].tolist()
    }
    response = HttpResponse(json.dumps(response))
    response["Access-Control-Allow-Origin"] = "*"
    return response


@csrf_exempt
def cam(request):
    image = request.FILES['file']
    temp, temp_name = tempfile.mkstemp()
    for chunk in image.chunks():
        os.write(temp, chunk)

    model = load_model()
    cam = GradCAM(model=model, target_layer=model.layer4[-1])
    rgb_img = cv2.imread(
        temp_name, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = image_to_tensor(rgb_img.copy(), resize=False)
    grayscale_cam = cam(input_tensor=input_tensor,
                        eigen_smooth=True,
                        aug_smooth=True)

    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = Image.fromarray(cam_image, 'RGB')
    buffer = io.BytesIO()
    cam_image.save(buffer, format="PNG")
    response = HttpResponse(buffer.getvalue())
    response["Access-Control-Allow-Origin"] = "*"
    return response
