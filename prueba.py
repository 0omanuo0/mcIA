import torch
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
import timeit
import matplotlib.pyplot as plt
from utils.visualize import get_color_pallete
import torch
from torch import nn


class SCNN:
    def __init__(self) -> None:
        self.dataset = 'citys'
        self.weights_folder = './weights'
        self.cpu = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_fast_scnn(self.dataset, pretrained=True, root=self.weights_folder, map_cpu=self.cpu).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(self.device)

    def segment(self, image):
        # image transform
        conv = transforms.ToTensor()
        image = self.transform(conv(image).to(self.device)).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, self.dataset)
        return mask
        #mask = get_color_pallete(pred, dataset)
