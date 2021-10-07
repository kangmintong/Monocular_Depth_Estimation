import torch
from models import ResNet
from PIL import Image
import transforms
import numpy as np
import cv2
import os
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.viridis

iheight, iwidth = 480, 640 # raw image size

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, transform=None, target_transform=None):
        self.input_path=input_path
        if not isinstance(input_path,list):
            self.input_path=[]
            self.input_path.append(input_path)
        self.target_transform=target_transform
        self.output_size = (228, 304)

    def __getitem__(self, index):
        img = Image.open(self.input_path[index]).convert('RGB')
        img = self.val_transform(img)
        #img=torch.cat([torch.tensor(img,dtype=torch.float),torch.rand((1,228,304),dtype=torch.float)],0)
        img=torch.tensor(img,dtype=torch.float)
        return img

    def __len__(self):
        return len(self.input_path)

    def val_transform(self, rgb):
        rgb=np.array(rgb)
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor(),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        return rgb_np


def create_eval_loader(input_path):

    mydataset=EvalDataset(input_path)
    return torch.utils.data.DataLoader(mydataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=None,)


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def display(depth_pred):
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    d_min = np.min(depth_pred_cpu)
    d_max = np.max(depth_pred_cpu)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    return depth_pred_col

def evaluate_single_img(model_path, input_path, output_dir):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model=checkpoint['model']
    model.eval()
    eval_loader=create_eval_loader(input_path)

    for idx, img in enumerate(eval_loader):
        output=model(img)
        output= output.clone().detach()
        output=display(output)
        output = Image.fromarray(output.astype('uint8'))
        output.save(os.path.join(output_dir,'output.jpg'))


if __name__=='__main__':
    evaluate_single_img('/Users/kmt/Desktop/Monocular_Depth_Estimation/results/nyudepthv2.sparsifier=uar.samples=0.modality=rgb.arch=resnet50.decoder=deconv3.criterion=l1.lr=0.01.bs=8.pretrained=True/model_best.pth.tar',
                        '/Users/kmt/Desktop/Monocular_Depth_Estimation/Image-labeling-examples-on-NYU-Depth-v2-dataset-Left-to-right-RGB-image-depth.png',
                        '/Users/kmt/Desktop/Monocular_Depth_Estimation/')