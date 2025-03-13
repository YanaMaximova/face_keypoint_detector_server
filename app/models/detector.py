import os
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import FaceModel

def detect(model_path, filename):
    model_test = FaceModel.load_from_checkpoint(checkpoint_path=model_path)
    model_test.to('cpu')
    model_test.eval()
    #imgs_names = os.listdir(test_img_dir)
    predictions = {}

    image = Image.open(filename).convert("RGB")
    image = np.array(image)
    normolize = A.Compose(
        [
            A.Resize(284, 284),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    norm_image = normolize(image=image)['image']

    pred = model_test(norm_image[None, ...].to('cpu')).reshape(-1)
    pred[0::2] *= image.shape[0] / 284
    pred[1::2] *= image.shape[1] / 284
    predictions[filename] = np.round(np.array(pred.detach().cpu().numpy()))
    return np.round(np.array(pred.detach().cpu().numpy()))