#
import os
import json
import random

from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset,DataLoader



class AlgaeDataset(Dataset):
    def __init__(self,configure,trained,extend=5):

        self.trained = trained
        self.configure =configure
        self.base_dir = self.configure["base_dir"]
        self.image_path = self.get_image_path()
        self.marks = self.configure['species']
        self.extend = extend

    def __len__(self):

        return len(self.image_path)*self.extend

    def __getitem__(self, item):
        for idx in range(self.extend):
            path = self.image_path[item]
            image = self.read_image(path)
            transformer = self.get_transforms()
            mark = path.split("\\")[-2]
            label = self.marks[mark]

            return transformer(image),label

    def get_transforms(self):

        transformers = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomHorizontalFlip(p=0.8),
            v2.RandomHorizontalFlip(p=0.8),
            v2.ColorJitter(brightness=0.5,contrast=0.5),
            v2.RandomGrayscale(p=0.1),
            v2.RandomRotation(360),]

        algae_trans = [v2.Resize((224,224))] + random.choices(transformers,k=2)+[v2.ToTensor(),v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]

        return v2.Compose(algae_trans)


    def read_image(self,path):
        return Image.open(path).convert("RGB")


    def get_image_path(self):

        image_path = []
        data_dirs = [os.path.join(self.base_dir,dir) for dir in os.listdir(self.base_dir)]
        for data_dir in data_dirs:
            for idx,path in enumerate(os.listdir(data_dir)):
                config_idx = self.configure['config']['train_index']
                if self.trained:
                    if idx < config_idx:
                        image_path.append(os.path.join(data_dir, path))

                else:
                    if idx >= config_idx:
                        image_path.append(os.path.join(data_dir,path))

        return image_path



def get_trainData():
    with open("configure.json", 'r') as fp:
            algae_data = AlgaeDataset(json.load(fp),True,extend=1)
            dataloader = DataLoader(algae_data,batch_size=1,shuffle=True)
            return dataloader
def get_testData():
    with open("configure.json", 'r') as fp:
            algae_data = AlgaeDataset(json.load(fp),False,extend=1)
            dataloader = DataLoader(algae_data,batch_size=1,shuffle=True)
            return dataloader
