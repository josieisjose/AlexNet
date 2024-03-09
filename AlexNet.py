import torch
import torchvision
import  json
import argparse
import logging
import os
import torch.nn as nn
from torch.utils.tensorboard import  SummaryWriter
import torchvision.datasets
from torch.utils.data import DataLoader
#参数配置
parser=argparse.ArgumentParser()
parser.add_argument("--device",type=str,default='cuda',help='choose cpu or cuda')
parser.add_argument("--output_dir",type=str,default="AlexNet/runs")
parser.add_argument("--epoch",type=int,default=20,help="the number of epoch")
parser.add_argument("--batch_size",type=int,default=8,help="the size of batch")
parser.add_argument("--lr",type=float,default=0.01,help="learning rate")
parser.add_argument("--momentum",type=float,default=0.5,help="momentum")
parser.add_argument("--dropout",type=float,default=0.5,help="dropout rate")
opt=parser.parse_args()
print(opt)

#创建输出文件
dir=os.makedirs(f'{opt.output_dir}',exist_ok=True)
#cpu/gpu
if opt.device:
    use_device="cuda" if torch.cuda.is_available() else "cpu"

#设置logs函数
def get_logger(filename,verbosity=1,name=None):
    "verbosity是日志详细程度，name指logger的名字"
    level_dict={0: logging.DEBUG,1: logging.INFO,2: logging.WARNING}
    formatter=logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s]%(message)s")#日志记录格式化器，指定记录格式
    logger=logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh=logging.FileHandler(filename,'w')#将日志写入指定文件
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()#将日志输出到控制台
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
log=get_logger(f'{opt.output_dir}/exp.txt')

#设置参数输出文件
content=opt
content=vars(content)#转化成json的序列化对象
config=os.path.join(opt.output_dir,'config.json')
with open(config,'w',encoding='utf-8') as f:
    json.dump(content,f,indent=4,ensure_ascii=False)

#数据集
train_data=torchvision.datasets.MNIST(root='data',train=True,transform=torchvision.transforms.Compose([torchvision.transforms.Resize(224),torchvision.transforms.ToTensor()]),download=False)
val_data=torchvision.datasets.MNIST(root='data',train=False,transform=torchvision.transforms.Compose([torchvision.transforms.Resize(224),torchvision.transforms.ToTensor()]),download=False)
train=DataLoader(dataset=train_data,batch_size=opt.batch_size,shuffle=True)
val=DataLoader(dataset=val_data,batch_size=opt.batch_size,shuffle=True)

#网络
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        #输入1*224*224
        self.conv1=nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)#96*54*54
        self.pool1=nn.MaxPool2d(3,2)#96*26*26
        self.conv2=nn.Conv2d(96,256,5,padding=2)#256*26*26
        self.pool2=nn.MaxPool2d(3,2)#256*12*12
        self.conv3=nn.Conv2d(256,384,3,padding=1)#384*12*12
        self.conv4=nn.Conv2d(384,384,3,padding=1)#384*12*12
        self.conv5=nn.Conv2d(384, 256, 3, padding=1)  # 256*12*12
        self.pool3=nn.MaxPool2d(3,2)#256,5,5
        self.flat=nn.Flatten()
        self.dropout=nn.Dropout(p=opt.dropout)
        self.linear1=nn.Linear(256*5*5,4096)
        self.linear2=nn.Linear(4096,4096)
        self.linear3=nn.Linear(4096,10)

    def forward(self,x):
        x=self.pool1(nn.ReLU()(self.conv1(x)))
        x=self.pool2(nn.ReLU()(self.conv2(x)))
        x=nn.ReLU()(self.conv3(x))
        x=nn.ReLU()(self.conv4(x))
        x=nn.ReLU()(self.conv5(x))
        x=self.pool3(x)
        x=self.flat(x)
        x=nn.ReLU()(self.linear1(x))
        x=self.dropout(x)
        x=nn.ReLU()(self.linear2(x))
        x =self.dropout(x)
        x=self.linear3(x)
        return x

#实例化网络
net=AlexNet().to(torch.device(use_device))
#损失函数
loss_func=torch.nn.CrossEntropyLoss()
#优化器
optim=torch.optim.SGD(net.parameters(),lr=opt.lr,momentum=opt.momentum)
#画图
writer1=SummaryWriter('AlexNet/runs/logs')
writer2=SummaryWriter('AlexNet/runs/logs')
#训练
def trainer(epoch):
    run_loss=0
    right=0
    total_samples=0
    for i,data in enumerate(train,1):
        x,y=data
        x=x.to(torch.device(use_device))
        y=y.to(torch.device(use_device))
        y_hat=net(x)
        cost=loss_func(y_hat,y)
        optim.zero_grad()
        cost.backward()
        optim.step()
        with torch.no_grad():
            run_loss += cost.data.sum()  # 计算当前迭代的损失
            right+=(torch.argmax(y_hat,dim=1)==y).sum().item()#计算当前迭代的正确率
            total_samples += y.size(0)
    train_loss=run_loss/len(train)
    train_acc=right/total_samples*100
    log.info(f'Epoch[{epoch+1}/{opt.epoch}]\t [train]\t loss={train_loss:.3f}\t acc={train_acc:.3f}%')
    return train_loss,train_acc
def valer():
    val_loss=0
    val_right=0
    total_samples=0
    with torch.no_grad():
        for i,data in enumerate(val,1):
            x,y=data
            x=x.to(torch.device(use_device))
            y=y.to(torch.device(use_device))
            y_hat=net(x)
            cost=loss_func(y_hat,y)
            val_loss+=cost.data.sum()
            val_right+=(torch.argmax(y_hat,dim=1)==y).sum().item()
            total_samples += y.size(0)
        val_loss=val_loss/len(val)
        val_acc=val_right/ total_samples *100
        log.info(f'[val]\t loss={val_loss:.3f}\t acc={val_acc:.3f}%')
    return val_loss,val_acc
#开始
log.info("start training !")
for epoch in range(opt.epoch):
    train_loss,train_acc=trainer(epoch)
    val_loss,val_acc=valer()
    writer1.add_scalars('acc',{'train_acc':train_acc,'val_acc':val_acc},epoch+1)
    writer2.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch + 1)
pt=torch.save(net.state_dict(),'AlexNet/runs/net.pt')
net_copy=AlexNet().to(torch.device(use_device))
net_copy.load_state_dict(torch.load('AlexNet/runs/net.pt'))