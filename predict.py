import random
import os
import json
from PIL import Image
from PIL.ImageFilter import GaussianBlur
import torch
from hashlib import md5
from torchvision.transforms import v2
from models import AlgaeYOLO,AlgaeSTYolo

from matplotlib import pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def blur_now(image:Image):
    return image.filter(GaussianBlur(radius=3))

def gray_now(image):
    return image.convert('L')

def get_image(algae_type,blur,gray):
    path = os.path.join("./data",algae_type)
    num = random.choices(["0"+str(n) for n in range(1,10)]+[str(n) for n in range(11,61)])
    image_path = os.path.join(path,str(num[0])+".jpg")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    if gray == True:
        image = gray_now(image)
    if blur == True:
        image = blur_now(image)

    return image

def get_transformer():
    transformers = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomHorizontalFlip(p=0.8),
            v2.RandomHorizontalFlip(p=0.8),
            v2.ColorJitter(brightness=0.5, contrast=0.5),
            v2.RandomGrayscale(p=0.1),
            v2.RandomRotation(360), ]

    algae_trans = [v2.Resize((224, 224))] + random.choices(transformers, k=2) + [v2.ToTensor(), v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return v2.Compose(algae_trans)

def predict(model,name,blur,gray):

    image_data = get_image(name,blur,gray)
    transformer = get_transformer()
    data = transformer(image_data)
    data = data.view(1,3,224,224).cuda()
    prediction = model(data)
    _, pred = torch.max(prediction.data, dim=1)
    show(image_data,pred)


def show(image,label):

    with open("configure.json",'r') as fi:
        labels = json.load(fi)
        for i,k in labels["species"].items():
            if k == int(label.data):
                plt.figure(i)
                plt.title(i)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(image)
                plt.savefig('./res/{}.jpg'.format(i))
                plt.close()





# algae = torch.load("models/model.pt")
#
# algae.eval()
# with open("configure.json",'r') as fi:
#     names = json.load(fi)
#     for name in names["species"].keys():
#         predict(algae,name,blur=False,gray=True)


def multiple_predict():
    algae_yolo = AlgaeYOLO('models/simple.pt')
    base_path = "temp"
    for i,pic in enumerate(os.listdir(base_path)):
        path = os.path.join(base_path,pic)
        algae_yolo.predict(path,"res/multi_{}.jpg".format(i))

def simple_predict(path):
    algae_yolo = AlgaeYOLO('models/simple.pt')
    algae_yolo.predict(path, "res/{}.jpg".format(md5(path.split("/")[-1].encode('utf-8')).hexdigest()))

multiple_predict()