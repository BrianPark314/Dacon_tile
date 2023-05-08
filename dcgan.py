#%%
# F5 대신 퍼센트 위의 런셀 누르면 새로운 창에서 실행되어 plt 출력그림이 나옵니다
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
from pathlib import Path
#plt.ion()

torch.manual_seed(42) #파이토치 시드 고정
#device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

print (f"PyTorch version:{torch.__version__}") # 1.12.1 이상
print(f"MPS device built: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS device available: {torch.backends.mps.is_available()}") # True 여야 합니다.

args = easydict.EasyDict()
args.workers = 2
args.image_size = 64
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
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# prepare datas        
def prep():
    print('='*50)
    print('Pytorch DCGAN loaded with pre-trained parameters')

    dataroot = str(Path('data/train'))
    dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.BATCH_SIZE,
                                         shuffle=True, num_workers=args.workers)

    #train_data, validation_data, test_data = engine.get_data(args.BATCH_SIZE, args.transform)
    print('Data preperation complete.')
    print('='*50)

    '''# #%%에서 Run cell 하면 64개 이미지 출력
    real_batch = next(iter(dataloader))[0]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
'''
    return dataloader


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



if __name__ == '__main__':
    #train_data, validation_data, test_data, dataloader = prep()
    dataloader = prep()
    #results = go(model, train_data, validation_data)

    # Create Generator
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init) # 가중치 초기화
    # print generator structure
    print('\n','='*50)
    print('Generator net Structure . . .')
    print(netG)
    print('='*50,'\n')

    # Create Discriminator net
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init) # 가중치 초기화
    # print discriminator structure
    print('\n','='*50)
    print('Discriminator net Structure . . .')
    print(netD)
    print('='*50,'\n')

########################################################33
##############################################################

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
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


    '''
    구분자 : log(D(x))+log(1−D(G(z))) 를 최대화
    생성자 : log(1−D(G(z))) 를 최소화

    Train
    '''

    # Create empty results dictionary
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    #train
    print("Starting Training Loop...")
        # 에폭(epoch) 반복
    for epoch in tqdm(range(args.NUM_EPOCHS)):
        # 한 에폭 내에서 배치 반복
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
            ###########################
            '''진짜 데이터들로 학습을 합니다'''
            netD.zero_grad()
            # 배치들의 사이즈나 사용할 디바이스에 맞게 조정합니다
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label,
                            dtype=torch.float, device=device)
            # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
            output = netD(real_cpu).view(-1)
            # 손실값을 구합니다
            errD_real = criterion(output, label)
            # 역전파의 과정에서 변화도를 계산합니다
            errD_real.backward()
            D_x = output.mean().item()

            '''가짜 데이터들로 학습을 합니다'''
            # 생성자에 사용할 잠재공간 벡터를 생성합니다
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # G를 이용해 가짜 이미지를 생성합니다
            fake = netG(noise)
            label.fill_(fake_label)
            # D를 이용해 데이터의 진위를 판별합니다
            output = netD(fake.detach()).view(-1)
            # D의 손실값을 계산합니다
            errD_fake = criterion(output, label)
            # 역전파를 통해 변화도를 계산합니다. 이때 앞서 구한 변화도에 더합니다(accumulate)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다
            # 이때 errD는 역전파에서 사용되지 않고, 이후 학습 상태를 리포팅(reporting)할 때 사용합니다
            errD = errD_real + errD_fake
            # D를 업데이트 합니다
            optimizerD.step()

            ############################
            # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
            # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
            # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
            output = netD(fake).view(-1)
            # G의 손실값을 구합니다
            errG = criterion(output, label)
            # G의 변화도를 계산합니다
            errG.backward()
            D_G_z2 = output.mean().item()
            # G를 업데이트 합니다
            optimizerG.step()

            '''print train status '''
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.NUM_EPOCHS, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # 이후 그래프를 그리기 위해 손실값들을 저장해둡니다
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # fixed_noise를 통과시킨 G의 출력값을 저장해둡니다
            if (iters % 500 == 0) or ((epoch == args.NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())


    '''
    - 19개로 분류되는지 확인하기
    - 
    '''