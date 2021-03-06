import io
import json

import cv2
import numpy as np
from PIL import Image
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torch.autograd import Variable
from .utils import image_to_tensor, get_image_from_request, load_model


def predict(model, image):
    model.eval()
    image_tensor = image_to_tensor(image)
    inp = Variable(image_tensor)
    output = model(inp)
    m = nn.Softmax(dim=1)
    return m(output)


@csrf_exempt
def classify(request):
    rgb_img = get_image_from_request(request)
    image = Image.fromarray(rgb_img, 'RGB')
    model = load_model()
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
    rgb_img = get_image_from_request(request)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = image_to_tensor(rgb_img.copy(), resize=False)
    model = load_model()
    cam = GradCAM(model=model, target_layer=model.layer4[-1])
    grayscale_cam = cam(input_tensor=input_tensor,
                        eigen_smooth=True,
                        aug_smooth=True)

    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = Image.fromarray(cam_image, 'RGB')
    buffer = io.BytesIO()
    cam_image.save(buffer, format="PNG")
    response = HttpResponse(buffer.getvalue())
    response["Access-Control-Allow-Origin"] = "*"
    return response
