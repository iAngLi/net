import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from operator import itemgetter, attrgetter

from ResNet import ResNet18

from MyTestData import MyDataset

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 135  # 遍历数据集次数   135
pre_epoch = 76  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testset = MyDataset('./data/test', train = False);
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
# Cifar-10的标签
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)
net_path = os.path.join(args.outf, 'net_final.pth')
net.load_state_dict(torch.load(net_path, map_location='cpu'))

# 训练
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # 2 初始化best test accuracy
    print("Start testing, Resnet-18!")  # 定义遍历数据集的次数
    result = []
    with open("result.txt", "w") as f:
        # 全部训练完打印label
        print("Waiting Test!")
        with torch.no_grad():
            for data in testloader:
                net.eval()
                images, labels = data
                images = images.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(len(labels)):
                    b= labels[i].split('.')[0]#不带后缀的文件名
                    result.append((int(b),int(predicted[i].item())))
                                   
        result = sorted(result, key=itemgetter(0), reverse=False)                  
        for i in range(len(result)):
            f.write("%d.png  %d\n" % (result[i][0], result[i][1]))
            f.flush()
        f.close()
    print("Test Finished")
