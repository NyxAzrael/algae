


from models import get_resnet50,AlgaeTransformer,AlgaeSwinTransformer,AlgaeSTYolo
from torchvision import models
from Dataset import get_trainData,get_testData

from engine import Engine

import torch.nn as nn
from torch.optim import SGD,AdamW



algae = AlgaeSwinTransformer()

Ioss = nn.CrossEntropyLoss()
optimizer = AdamW(algae.parameters(),lr=0.0001,weight_decay=0.01)

engine = Engine(model=algae,
                optimizer=optimizer,
                criterion=Ioss,
                train_dataloader=get_trainData(),
                test_dataloader=get_testData(),
                scanshot_path=".\\models\\scanshot.pt")

engine.train(20,save_scanshot=False,saved=False)

def yolo_train():
    model = AlgaeSTYolo()
    results = model.train(data="data.yaml", epochs=120, batch=8, imgsz=640)