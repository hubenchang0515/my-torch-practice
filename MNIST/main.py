#! /usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 配置
batch_size:int = 64     # 一次处理多少数据
epochs:int = 5          # 总共学习多少轮

# 判断是否支持GPU加速(pytorch里用代指所有GPU)
device:str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# 定义神经网络模型
# 使用了这里的模型 https://zhuanlan.zhihu.com/p/340379189
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 神经网络
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),         # 二维卷积，输入为28x28x1，卷积核宽度为3，因此输出为26x26x32（输出channels可以随意设置）
            nn.ReLU(),                      # 线性激活
            nn.Conv2d(32, 64, 3, 1),        # 二维卷积，输入为26x26x32，卷积核宽度为3，因此输出为24x24x64
            nn.ReLU(),
            nn.MaxPool2d(2),                # 二维最大池化（采样），输出为12x12x64
            nn.Dropout(0.25),               # 随机忽略一定比例的神经元，避免过拟合
            nn.Flatten(),                   # 折叠 12x12x64 -> 9216
            nn.Linear(9216, 128),           # 全连接 9216 -> 128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)              # 全连接 128 -> 10
        )

    # 正向传导
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# 定义训练函数
def train(dataloader:DataLoader, model:NeuralNetwork, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # 开始训练
    for batch, (X, y) in enumerate(dataloader):
        # 取出一组数据并移动到对应的设备上
        X, y = X.to(device), y.to(device)
        
        # 使用模型进行一次预测，并计算损失
        prediction = model(X) 
        loss = loss_fn(prediction, y)

        # 进行反向传播，对模型进行修正
        optimizer.zero_grad()   # 梯度置0
        loss.backward()         # 反向传播
        optimizer.step()        # 更新参数

        # 打印训练过程
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 定义测试函数
def test(dataloader:DataLoader, model:NeuralNetwork, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval() # 开始计算
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) 

            # 使用模型进行一次预测，并计算损失
            prediction = model(X)                                                   # 生成预测函数
            test_loss += loss_fn(prediction, y).item()                              # 计算损失
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()   # 计算结果并与标签进行比对

    test_loss /= num_batches # 平均损失
    correct /= size          # 正确率
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



# 入口函数
def main():
    # 获得数据集
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # 创建DataLoader,每次取出 batch_size 个数据
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # 打印输入输出张量的形状
    for X, y in test_dataloader:
        print("Input tensor shape[N, C, H, W]: ", X.shape)   # 输入 batch_size 个 1张 28x28的图片
        print("Output tensor shape: ", y.shape, y.dtype)       # 输出 batch_size 个 图片对应的数字
        break

    # 创建模型
    model = NeuralNetwork().to(device)
    print(model)

    # 损失函数，用来表示估计函数与训练集之间的误差的函数
    loss_fn = nn.CrossEntropyLoss()

    # 参数优化器，使用AdaDelta算法
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    # 进行训练和测试
    for i in range(epochs):
        print(f"epoch {i}")
        train(train_dataloader, model, loss_fn, optimizer)  # 训练
        test(test_dataloader, model, loss_fn)               # 测试
    

if __name__ == "__main__":
    main()