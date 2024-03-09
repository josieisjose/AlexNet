# AlexNet
## 说明  
  本代码是在看完李沐老师的《动手深度学习Pytorch篇》+相应网络论文后，进行的网络复现；  
  欢迎各位使用并指出不足🫡  
## 环境依赖  
* python=3.8  
* torch=2.0.0  
* torchvision=0.15.0
* tensorboard=2.14.0
## 目录结构
```
│  README.md          //帮助文档
│  AlexNet.py  
├─data                //下载or存放数据集的文件 
│  │          
│  └─MNIST  
│      └─raw  
│              
├─AlexNet               //存放结果的文件
│  └─runs  
│     └─logs      //存放tensorboard绘制的图片的文件夹
│     └─config.json      //保存的配置文件
│     └─exp.txt          //生成的训练日志
      └─net.pt           //保存的权重       
```
## 使用说明  
1. 读取数据、构建网络等部分都放在了一个文件中（即AlexNet.py）
2. 使用时只需要修改部分参数和配置
3. 默认使用MNIST数据集
4. 默认使用GPU进行训练
5. 本代码可视化使用的是torch.utils.tensorboard  
6. 由于网络变深但数据集简单，**因此训练20轮时出现了过拟合，使用者可以把epoch调小并适当减小学习率；**
## 结果展示  
### 准确率 
![image](https://github.com/josieisjose/AlexNet/blob/main/AlexNet/acc.png)
### 损失   
![image](https://github.com/josieisjose/AlexNet/blob/main/AlexNet/loss.png)
