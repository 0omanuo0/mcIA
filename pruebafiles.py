import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete
import timeit
import numpy as np 

outdir = './out'
input_pic = './IMG_4459.jpg'
dataset = 'citys'
weights_folder = './weights'
cpu = None

def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # output folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(input_pic).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model = get_fast_scnn(dataset, pretrained=True, root=weights_folder, map_cpu=cpu).to(device)
    print('Finished loading model!')
    t1 = timeit.default_timer()
    model.eval()
    t2 = timeit.default_timer()
    with torch.no_grad():
        for i in range(100):
            outputs = model(image)
    t3 = timeit.default_timer()
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    t4 = timeit.default_timer()
    plt.imshow(pred)
    plt.show()
    mask = get_color_pallete(pred, dataset)
    print(np.array(mask.convert('RGB')).shape)
    print(t2-t1, t3-t2, t4-t3, t4-t1)
    outname = os.path.splitext(os.path.split(input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(outdir, outname))


if __name__ == '__main__':
    demo()
