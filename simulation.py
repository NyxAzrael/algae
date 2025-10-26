import matplotlib.pyplot as plt
from PIL import Image
import random
from tools import blur_now



class Zooplankton():
    def __init__(self, name, init_position,size):
        self.name = name
        self.size = size
        self.position = init_position


    def init_image(self,src):
        angle = [random.randint(0,360) for _ in range(40)]
        random.shuffle(angle)
        return blur_now(Image.open(src).rotate(random.choice(angle)))

    def random_move(self, x, y):
        if x > 1 or x < 0 or y > 1 or y < 0:
            x = 0
            y = 0
        foot = [-0.01 for _ in range(50)] + [0.01 for _ in range(20)]+ [random.uniform(-0.01,0.01) for _ in range(40)]
        random.shuffle(foot)
        return x + random.choice(foot), y + random.choice(foot)



    def get_position(self):
        x, y = self.position
        x,y = self.random_move(x, y)
        self.position = (x,y)
        return x,y



def init_background(figure):
    back_pic = Image.open("basesrc/back.jpg")
    background = figure.add_axes([0,0,1,1])
    background.imshow(back_pic)


def sim_creator(plankton,times):

    fig = plt.figure(figsize=(6.4,6.4))

    for n in range(times):
        init_background(fig)
        for plk in plankton:

            x,y = plk.get_position()
            ax = fig.add_axes([x, y, plk.size, plk.size])
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            image = plk.init_image(plk.name)
            ax.imshow(image)

        plt.savefig("temp/{}.jpg".format(n))
        fig.clf()

plankton = [
            Zooplankton("basesrc/2.png",(0.1,0.5),0.2),

           ]

sim_creator(plankton,6)