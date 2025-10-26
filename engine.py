import os.path
import json
import os
import torch
import torch.nn as nn
from numexpr.expressions import double
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

class Engine():
    def __init__(self,model:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 criterion:nn.CrossEntropyLoss,
                 train_dataloader:DataLoader,
                 test_dataloader:DataLoader,
                 scanshot_path:str):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.gpu = torch.device("cuda:0")
        self.configure = self.__get_configure()
        self.scanshot_path = scanshot_path
        self.now_epoch = 0
        self.train_loss,self.test_loss = [],[]
        self.train_accuracy,self.test_accuracy = [],[]
        self.best_accuracy = 0


    def __get_configure(self):
        with open("configure.json", 'r') as fp:
            return json.load(fp)


    def __run_batch(self,train_data,train_label,eval = False):
        self.optimizer.zero_grad()
        pre_data = self.model(train_data)
        loss = self.criterion(pre_data,train_label)
        loss.backward()
        self.optimizer.step()
        if not eval:
            self.train_accuracy.append(self.__accurate(pre_data,labels=train_label))
            self.train_loss.append(loss.item())
        else:
            self.test_accuracy.append(self.__accurate(pre_data,labels=train_label))
            self.test_loss.append(loss.item())



    def __run_epoch(self):
        for train_data,train_label in self.train_dataloader:
            train_data,train_label = train_data.to(self.gpu),train_label.to(self.gpu)
            self.__run_batch(train_data,train_label)


    def __save(self):
        if not os.path.exists("./models"):
            os.mkdir("./models")
        torch.save(self.model,'./models/model.pt')
        # print("模型成功保存在./models/model.pt")

    def __save_scanshot(self,epoch):
        scanshot = {}
        scanshot["MODEL_STATE"] = self.model.state_dict()
        scanshot["EPOCH"] = epoch
        if not os.path.exists("./models"):
            os.mkdir("./models")
        torch.save(scanshot,"./models/scanshot.pt")
        print("保存快照到./models/scanshot.pt")

    def __load_scanshot(self,scanshot_path):
        scanshot = torch.load(scanshot_path)
        self.now_epoch = scanshot["EPOCH"]
        if self.now_epoch == self.configure['config']['epochs']:
            self.now_epoch = 0
            os.remove("./models/scanshot.pt")
            print("上次已经完成，将删除快照重新运行")
        else:
            self.model.load_state_dict(scanshot["MODEL_STATE"],strict=False)
        print("成功加载最近一次快照，继续训练......")

    def __accurate(self,pre_data,labels):
        pre, target = pre_data.cpu(),labels.cpu()
        _, pred = torch.max(pre.data, dim=1)
        rights = torch.eq(pred,target).sum()
        return 100 * rights / len(target)

    def eval(self):
        self.model.eval()
        for test_data,test_label in self.test_dataloader:
            test_data,test_label = test_data.to(self.gpu),test_label.to(self.gpu)
            self.__run_batch(test_data,test_label,eval = True)
        test_loss = sum(self.test_loss)/len(self.test_loss)
        test_accuracy = sum(self.test_accuracy) / len(self.test_accuracy)
        return test_loss,test_accuracy

    def train(self,epochs:int,save_scanshot,saved):
        if os.path.exists(self.scanshot_path) and save_scanshot:
            self.__load_scanshot(self.scanshot_path)
        self.model.to(self.gpu)
        self.model.train()
        for epoch in range(self.now_epoch,epochs+1):
            self.__run_epoch()
            if epoch%self.configure['config']["save_scanshot"] == 0 and saved:
                self.__save_scanshot(epoch)
            msg = "[GPU##{}]  epoch:{}  train_loss:{:.3f}  train_accuracy:{:.3f}%  test_loss:{:.3f}  test_accuracy:{:.3f}%"
            train_loss = sum(self.train_loss) / len(self.train_loss)
            train_accuracy = sum(self.train_accuracy) / len(self.train_accuracy)
            test_loss, test_accuracy = self.eval()
            # print(msg.format(
            #     self.gpu,
            #     epoch,
            #     train_loss,train_accuracy,
            #     test_loss,test_accuracy,
            # ))

            print(double(test_loss))
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                self.__save()