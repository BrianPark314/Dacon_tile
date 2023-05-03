#%%
import torch
import torchvision
import time
import engine
from torch import nn
import easydict
import torchinfo
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
#plt.ion()

torch.manual_seed(42) #파이토치 시드 고정
#device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

print (f"PyTorch version:{torch.__version__}") # 1.12.1 이상
print(f"MPS device built: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS device available: {torch.backends.mps.is_available()}") # True 여야 합니다.

args = easydict.EasyDict()
args.workers = 2
args.BATCH_SIZE = 128
args.nc = 3 # RGB channels
args.nz = 100 # 잠재공간 벡터의 크기 (예. 생성자의 입력값 크기)
args.ngf = 64 # 생성자를 통과하는 특징 데이터들의 채널 크기
args.ndf = 64 #구분자를 통과하는 특징 데이터들의 채널 크기
args.NUM_EPOCHS = 10 
args.lr = 0.0002 # 옵티마이저의 학습률
args.beta1 = 0.5 # Adam Optimizer's beta1 hyper-parameter
args.transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 사용가능한 gpu 번호(1). CPU를 사용해야 하는경우 0으로 설정하세요
ngpu = 0

# 
class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule, self).__init__()
        self.layer1 = nn.Linear(1000, 19)
        self.net = torchvision.models.vgg16(
            weights = torchvision.models.VGG16_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        return (self.layer1(self.net(x)))


# prepare datas        
def prep():
    print('='*50)
    model = ClassifierModule()
    print('Pytorch DCGAN loaded with pre-trained parameters')

    model.to(device)
    
    train_data, validation_data, test_data = engine.get_data(args.BATCH_SIZE, args.transform)
    print('Data preperation complete.')
    
    print('='*50)

    # #%%에서 Run cell 하면 64개 이미지 출력
    real_batch = next(iter(train_data))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    return model, train_data, validation_data, test_data

# DCGAN 논문에서는, 평균이 0이고 분산이 0.02인 정규분포을 이용해, 구분자와 생성자 모두 무작위 초기화를 진행하는 것이 좋다고 합니다. 
# netG, netD에 적용할 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
# Generator(생성자)
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d( args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. (ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create Generator
netG = Generator(ngpu).to(device)
# 가중치 초기화
netG.apply(weights_init)
# print generator structure
print('\n','='*50)
print('Generator net Structure . . .')
print(netG)
print('='*50,'\n')


# Discriminator(구분자)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 (nc) x 64 x 64 입니다
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
# Create Discriminator net
netD = Discriminator(ngpu).to(device)
# 가중치 초기화
netD.apply(weights_init)
# print discriminator structure
print('\n','='*50)
print('Discriminator net Structure . . .')
print(netD)
print('='*50,'\n')



# 참 라벨 (혹은 정답)은 1로 두고, 거짓 라벨 (혹은 오답)은 0으로 둠 (GAN 구성 시의 관례)
# 두 옵티마이저는 모두 Adam을 사용하고, 학습률은 0.0002, Beta1 값은 0.5로 둠

# BCELoss 함수의 인스턴스를 생성합니다
criterion = nn.BCELoss()

# 생성자의 학습상태를 확인할 잠재 공간 벡터를 생성합니다
fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

# 학습에 사용되는 참/거짓의 라벨을 정합니다
real_label = 1.
fake_label = 0.

# G와 D에서 사용할 Adam옵티마이저를 생성합니다
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(args.beta1, 0.999))


'''
구분자 : log(D(x))+log(1−D(G(z))) 를 최대화
생성자 : log(1−D(G(z))) 를 최소화
'''





'''if __name__ == '__main__':
    model, train_data, validation_data, test_data = prep()
    #results = go(model, train_data, validation_data)'''
