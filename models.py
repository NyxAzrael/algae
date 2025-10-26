from torch.nn import functional as F
import torch
import timm
from torchvision import models
import torch.nn as nn
from torch.nn import Conv2d,Flatten,AvgPool2d,Linear,BatchNorm2d
from twisted.conch.insults.text import flatten
from ultralytics import YOLO
from PIL import Image



class AlgaeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = BatchNorm2d(num_features=3,momentum=0.1)
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.pool1 = AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.pool2 = AvgPool2d(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.linear = Linear(12*53*53,4)


    def forward(self,input):
        conv1 = self.conv1(input)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        output = self.flatten(pool2)



        return self.linear(output)




def get_resnet50():
    algae = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = algae.fc.in_features
    algae.fc = nn.Linear(num_ftrs, 5)

    for param in algae.parameters():
        param.requires_grad = False

    for param in algae.fc.parameters():
        param.requires_grad = True

    return algae

class AlgaeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.heads.parameters():
            param.requires_grad = True

    def get_model(self):
        num_ftrs = self.model.heads[0].in_features
        self.model.heads = nn.Linear(num_ftrs,5)
        self._freeze()

        return self.model


class AlgaeYOLO(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.model = YOLO(weights)

    def predict(self,images,path):
        res = self.model(images)

        for result in res:

            result.save(filename=path)


class AlgaeSwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.layers.head = nn.Linear(self.layers.head.in_features,24)
        self._freeze()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(24,5)


    def _freeze(self):

        # 冻结特征提取部分的所有层
        for param in self.layers.parameters():
            param.requires_grad = False

        # 解冻分类头层
        for param in self.layers.head.parameters():
            param.requires_grad = True

    def forward(self, input):

        features = self.layers(input)
      
        per = torch.permute(features,(0,3,1,2))
        pooled_features = F.adaptive_avg_pool2d(per, (1, 1))
        f =  self.flatten(pooled_features)
        output = self.linear(f)

        return output


class AlgaeSTYolo(YOLO):
    def __init__(self):
        super().__init__()
        self.yolo_layer = YOLO('yolo11n.pt')
        self.yolo_layer.model.model[0] = nn.Conv2d(24, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.swin_layers = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.swin_layers.head = nn.Linear(self.swin_layers.head.in_features, 24)
        self._freeze(self.swin_layers)

    def _freeze(self, model):

        # 冻结特征提取部分的所有层
        for param in model.parameters():
            param.requires_grad = False

        # 解冻分类头层
        for param in model.head.parameters():
            param.requires_grad = True

    def forward(self, input):
        swin = self.swin_layers(input)
        return self.yolo_layer(swin)