import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import Compose, ToTensor, Normalize


def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img.copy()).unsqueeze(0)


model = torch.load("/home/mardzin/dev/python/covid-detector-model/modelV2.pt")
# torch.jit.save(traced_model, "/home/mardzin/dev/python/covid-detector-model/modelV2-jit.pt")
cam = GradCAM(model=model, target_layer=model.layer4)

rgb_img = cv2.imread(
    "/home/mardzin/dev/python/covid-detector-model/COVID-19_Radiography_Dataset/test/covid/COVID-70.png", 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img)

grayscale_cam = cam(input_tensor=input_tensor,
                    target_category=None)

grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite(f'test_cam.jpg', cam_image)
