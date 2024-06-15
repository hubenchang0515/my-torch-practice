#! /usr/bin/env python3
import argparse
from datetime import datetime
import torch
from torch import nn
from torch.optim import Optimizer 
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 常量
AVATAR_HEIGHT:int = 64
AVATAR_WIDTH:int = 64
AVATAR_CHANNELS:int = 3
NOISE_CHANNELS:int = 100
REAL_LABEL:float = 0.9
FAKE_LABEL:float = 0.1

# 数据加载器
class AvatarDataLoader(DataLoader):
    def __init__(self, dir:str, batch_size:int):
        transform = transforms.Compose([
            transforms.Resize((AVATAR_HEIGHT, AVATAR_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.__dataset = datasets.ImageFolder(dir, transform=transform, target_transform=self.target_transform)
        super().__init__(self.__dataset, batch_size=batch_size)

    def __getitem__(self, index:int) -> torch.Tensor:
        return self.__dataset[index]
    
    def target_transform(self, _:any):
        return torch.tensor([REAL_LABEL], dtype=torch.float)
    
    def get_rand_batch(self) -> torch.Tensor:
        dataloader = DataLoader(self.__dataset, batch_size=self.batch_size)
        return next(iter(dataloader))

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        BASE_CHANNELS:int = 64

        self.model = nn.Sequential(
            nn.ConvTranspose2d(NOISE_CHANNELS, BASE_CHANNELS*8, 4, bias=False),
            nn.BatchNorm2d(BASE_CHANNELS*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(BASE_CHANNELS*8, BASE_CHANNELS*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(BASE_CHANNELS*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(BASE_CHANNELS*4, BASE_CHANNELS*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(BASE_CHANNELS*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(BASE_CHANNELS*2, BASE_CHANNELS, 4, 2, 1, bias=False),
            nn.BatchNorm2d(BASE_CHANNELS),
            nn.ReLU(True),
            nn.ConvTranspose2d(BASE_CHANNELS, AVATAR_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input) -> torch.Tensor:
        img = self.model(input)
        return img

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        BASE_CHANNELS:int = 64

        self.model = nn.Sequential(
            nn.Conv2d(AVATAR_CHANNELS, BASE_CHANNELS, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS*2, 4, 2, 1),
            nn.BatchNorm2d(BASE_CHANNELS*2), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(BASE_CHANNELS*2, BASE_CHANNELS*4, 4, 2, 1),
            nn.BatchNorm2d(BASE_CHANNELS*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(BASE_CHANNELS*4, BASE_CHANNELS*8, 4, 2, 1),
            nn.BatchNorm2d(BASE_CHANNELS*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(BASE_CHANNELS*8, 1, 4),
            nn.Sigmoid(),
        )

    def forward(self, input) -> torch.Tensor:
        logits = self.model(input)
        return logits
    
    def input_width(self) -> int:
        return AVATAR_WIDTH
    
    def input_height(self) -> int:
        return AVATAR_HEIGHT
    
class GAN(object):
    def __init__(self, 
                 device:str,
                 dataloader:DataLoader, 
                 generator:Generator, 
                 discriminator:Discriminator,
                 generator_optimizer:Optimizer,
                 discriminator_optimizer:Optimizer,
                 loss_fn:nn.Module,
                 ):
        super().__init__()
        self.__device:str = device
        self.__dataloader:DataLoader = dataloader
        self.__generator:Generator = generator.to(device=device)
        self.__discriminator:Discriminator = discriminator.to(device=device)
        self.__generator_optimizer:Optimizer = generator_optimizer
        self.__discriminator_optimizer:Optimizer = discriminator_optimizer
        self.__loss_fn:nn.Module = loss_fn.to(device=device)

    def device(self) -> str:
        return self.__device
    
    def dataloader(self) -> DataLoader:
        return self.__dataloader

    def batch_size(self) -> int:
        return self.__dataloader.batch_size

    def generate(self, batch_size:int=1) -> torch.Tensor:
        input:torch.Tensor = torch.randn(batch_size, NOISE_CHANNELS, 1, 1, device=self.__device)
        output:torch.Tensor = self.__generator(input)
        return output
    
    def mark(self, data:any, batch_size:int=1) -> torch.Tensor:
        label:torch.Tensor = torch.tensor(data, device=self.__device, dtype=torch.float)
        label:torch.Tensor = label.repeat(batch_size, 1)
        return label
    
    def generator_loss(self, image:torch.Tensor) -> torch.Tensor:
        batch_size:int = image.shape[0]
        want_label:torch.Tensor = self.mark([REAL_LABEL], batch_size=batch_size)
        prediction:torch.Tensor = self.__discriminator(image).view(batch_size, -1)
        loss: torch.Tensor = self.__loss_fn(prediction, want_label)
        return loss
    
    def discriminator_real_loss(self, real_image:torch.Tensor) -> torch.Tensor:
        batch_size:int = real_image.shape[0]
        want_label:torch.Tensor = self.mark([REAL_LABEL], batch_size=batch_size)
        prediction:torch.Tensor = self.__discriminator(real_image).view(batch_size, -1)
        loss: torch.Tensor = self.__loss_fn(prediction, want_label)
        return loss
    
    def discriminator_fake_loss(self, fake_image:torch.Tensor) -> torch.Tensor:
        batch_size:int = fake_image.shape[0]
        want_label:torch.Tensor = self.mark([FAKE_LABEL], batch_size=batch_size)
        prediction:torch.Tensor = self.__discriminator(fake_image).view(batch_size, -1)
        loss: torch.Tensor = self.__loss_fn(prediction, want_label)
        return loss
    
    def train_discriminator(self, real_image:torch.Tensor, fake_image:torch.Tensor):
        # self.__discriminator.train()
        self.__discriminator_optimizer.zero_grad()
        # self.__generator.eval() 不能使用 eval 模式，原因不明

        # 使用输入的图片训练判别器
        real_loss:torch.Tensor = self.discriminator_real_loss(real_image)
        real_loss.backward()

        # 使用生成的图片训练判别器
        # fake_image:torch.Tensor = self.generate(batch_size=self.batch_size())
        fake_loss_D:torch.Tensor = self.discriminator_fake_loss(fake_image.detach()) # 即使处于 eval 模式下也需要调用 detach
        fake_loss_D.backward()

        # 更新判别器权重
        self.__discriminator_optimizer.step()

    def train_generator(self, fake_image:torch.Tensor):
        # self.__generator.train()
        self.__generator_optimizer.zero_grad()
        # self.__discriminator.eval()

        # 生成图片，用判别器训练生成器
        # fake_image:torch.Tensor = self.generate(batch_size=self.batch_size())
        fake_loss_G:torch.Tensor = self.generator_loss(fake_image)
        fake_loss_G.backward()

        # 更新生成器权重
        self.__generator_optimizer.step()

    def train(self):
        for _, (real_image, _) in enumerate(self.__dataloader):
            real_image = real_image.to(self.__device)
            fake_image:torch.Tensor = self.generate(batch_size=self.batch_size())
            self.train_discriminator(real_image, fake_image)
            self.train_generator(fake_image)

# 设置模型权重的初始值，减少训练时间
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    parser = argparse.ArgumentParser(description="MNIST GAN Demo")
    parser.add_argument("-r", "--rate", help="learing rate", type=float, default=0.0002)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=128)
    parser.add_argument("-e", "--epoch", help="epoches", type=int, default=500)
    parser.add_argument("-m", "--model", help="model file", type=str, default="model.pt")
    parser.add_argument("-d", "--data", help="dataset path", type=str, default="./data/azur_lane_avatar")
    parser.add_argument("-g", "--generate", help="file of generating, None for training", type=str, default=None)
    args = parser.parse_args()


    try:
        gan:GAN = torch.load(args.model)
    except FileNotFoundError:
        device:str = "cuda" if torch.cuda.is_available() else "cpu"
        dataloader = AvatarDataLoader(args.data, batch_size=args.batch)
        generator = Generator()
        generator.apply(weights_init)
        discriminator = Discriminator()
        discriminator.apply(weights_init)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.rate, betas=(0.5, 0.999))
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.rate, betas=(0.5, 0.999))
        loss_fn = nn.BCELoss()
        gan:GAN = GAN(device=device, 
                      dataloader=dataloader,
                      generator=generator,
                      discriminator=discriminator,
                      generator_optimizer=generator_optimizer,
                      discriminator_optimizer=discriminator_optimizer,
                      loss_fn=loss_fn)
    
    if args.generate is None:
        G_loss_history:list[float] = []
        D_loss_history:list[float] = []

        for epoch in range(args.epoch):
            begin:datetime = datetime.now()
            gan.train()
            end:datetime = datetime.now()
            elapse:datetime = end - begin
            with torch.no_grad():
                # 生成预览图片
                fake_image = gan.generate(args.batch)
                (real_image, _) = gan.dataloader().get_rand_batch()
                G_loss = gan.generator_loss(fake_image)
                D_fake_loss = gan.discriminator_fake_loss(fake_image)
                D_real_loss = gan.discriminator_real_loss(real_image.to(device=gan.device()))
                D_loss = D_fake_loss + D_real_loss
                utils.save_image((fake_image + 1.0) / 2.0, "./preview.png")
                print(f"[{epoch+1}/{args.epoch}] elapse:{elapse} G_loss:{G_loss:#.6} D_loss:{D_loss:#.6}({D_real_loss:#.6} + {D_fake_loss:#.6}) ")

                # 生成 loss 曲线
                G_loss_history.append(G_loss.to("cpu"))
                D_loss_history.append(D_loss.to("cpu"))
                plt.clf()
                plt.plot(G_loss_history, "-", color="blue", label="G")
                plt.plot(D_loss_history, "-", color="green", label="D")
                plt.legend(loc=0)
                plt.savefig("./loss.png")

        torch.save(gan, args.model)

    else:
        # 生成一批图片，然后保存判别器认为最真的一张
        with torch.no_grad():
            image:torch.Tensor = gan.generate(args.batch)
            loss:torch.Tensor = gan.discriminator_fake_loss(image)
            best:torch.Tensor = (image[loss.argmax()] + 1.0) / 2.0
            utils.save_image(best, args.generate)

    
if __name__ == "__main__":
    main()