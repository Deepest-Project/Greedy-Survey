

```python
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
```

    Random Seed:  999





    <torch._C.Generator at 0x1f3cfb49a30>



- dataroot - 데이터셋 루트 폴더
- workers - DataLoader를 통해서 Data를 불러올 threads 숫자
- batch_size - 훈련에 쓸 배치 사이즈, DGGAN paper에서는 128을 씀.
- image_size - 훈련을 위해서 쓰인 이미지 사이즈. 이 코드에서는 default가 64x64이다. 바꾸려면 D, G 구조가를 바뀌야 한다.
- nc - input image의 채널 수
- nz - latent vector의 길이
- ngf - Generator의 feature map의 depth
- ndf - Discriminator의 feature map의 depth
- num_epochs - epochs 수
- lr - learning rate, DCGAN에서는 0.0002
- beta1 - Adam optimizer의 beta1, 0.5
- ngpu - GPUs 숫자, 0이면 cpu에서 돌아감.


```python
# Root directory for dataset
dataroot = "../data/celeba"
outf = "./results"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 20

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
```


```python
# We can use an image folder dataset the way we have it setup.
# Create the dataset


'''

torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)

torchvision.transforms.ToTensor
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes
    (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.

torchvision.transforms.functional.normalize(tensor, mean, std, inplace=False)
    Normalize a tensor image with mean and standard deviation.
    Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels,
    this transform will normalize each channel of the input torch.*Tensor
    i.e. input[channel] = (input[channel] - mean[channel])
'''
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
'''
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                            sampler=None, batch_sampler=None,
                            num_workers=0, collate_fn=<function default_collate>,
                            pin_memory=False, drop_last=False, timeout=0,
                            worker_init_fn=None)

shuffle: data를 각 epoch마다 다시 shuffle 할 것인지?
sampler (Sampler, optional) – defines the strategy to draw samples from the dataset. If specified, shuffle must be False.
    torch.utils.data.SequentialSampler(data_source)
    torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None)
    torch.utils.data.SubsetRandomSampler(indices)
    torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)
    torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
    torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None)
'''

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")

'''
torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False,
                            range=None, scale_each=False, pad_value=0)


'''
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```




    <matplotlib.image.AxesImage at 0x1f3d2565518>




![png](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/output_3_1.png?raw=true)


- Paper에서는 mean = 0.0, stdev=0.2의 Normal distribution을 사용했다고 함.


```python
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

이미지
- Generator는 latent vector(z)를 data-space 하도록 디자인 된다. 우리의 데이터는 images 이기 때문에 z를 data-space로 만든다는 것은 결국 training images(i.e. 3x64x64)를 만들어낸다는 것이다.
- strided 2D convolutional transpose layers - 2d batch norm layer - relu activation - tanh fuction [-1, 1].
- batch norm을 사용한 것이 DCGAN paper의 중요한 기여이다.


- torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

    The parameters kernel_size, stride, padding, dilation can either be
        - a single int
        - a tuple of two ints - height x width

- torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

    The parameters kernel_size, stride, padding, dilation can either be
        - a single int
        - a tuple of two ints - height x width

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/dcgan_generator.png?raw=true"  width="100%" height="100%">


```python
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nz = 100, ngf = 64
            # out_channel = 512
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/going_backward.png?raw=true"  width="100%" height="100%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/kernel.png?raw=true"  width="50%" height="0%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/convoultion_matrix.png?raw=true"  width="100%" height="100%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/going_backward.png?raw=true"  width="100%" height="100%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/output_with_conv_matrix.png?raw=true"  width="100%" height="100%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/output_with_transposed_conv_matrix.png?raw=true"  width="100%" height="100%">



출처: https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/


```python
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
```

    Generator(
      (main): Sequential(
        (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace)
        (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU(inplace)
        (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (11): ReLU(inplace)
        (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (13): Tanh()
      )
    )


- The DCGAN paper에서 pooling보다 strided convolution이 network가 own pooling function을 배우기 때문에 더 좋다고 말했다.



```python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```


```python
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
```

    Discriminator(
      (main): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): LeakyReLU(negative_slope=0.2, inplace)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): LeakyReLU(negative_slope=0.2, inplace)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): LeakyReLU(negative_slope=0.2, inplace)
        (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): LeakyReLU(negative_slope=0.2, inplace)
        (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (12): Sigmoid()
      )
    )


- real label: 1
- fake lable: 0
- set two separate optimizers, D, G.



```python
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

- we will “construct different mini-batches for real and fake” images, and also adjust G’s objective function to maximize logD(G(z))

- 2 main parts, 1 updates the Discriminator and 2 updates the generator.

### Part 1 - Train the Discriminator
- input이 real 인지 fake인지 분류하는 확률을 높이는 것이 목적이다.
- ascending its stochastic gradient로 discriminator를 update를 하길 원한다.
- Max log(D(x))+log(1−D(G(z)))

- 먼저, 실제 샘플을 훈련세트에서 만들고 D를 통해 forward를 한 후, loss를 계산한다. 그리고 backward pass의 gradient를 계산한다.
- 그리고 나서, fake samples을 생성하고 D를 통해 forward를 한 후, loss를 계산한다. 그 후, gradient를 accumulate한다.
- 마지막으로 acculmulated된 gradients를 가지고 D의 optimizer의 step를 실시한다.

### Part 2 - Train the Generator
- log(1−D(G(z))) 가 초기 단계에 gradient가 잘 안나옴.
- 그래서 maximize log(D(G(z)))를 하려고 함.
- How?
    - G의 output을 Part 1의 D를 통해서 classify하고, real labels을 GT로 해서 Generator's loss를 계산하고, backward pass의 gradients를 계산하고, optimizer step으로 업데이트한다.


```python
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
```
<details>
<summary>[0/20][0/1583]	Loss_D: 1.7410	Loss_G: 4.7765	D(x): 0.5343	D(G(z)): 0.5771 / 0.0136 <br />
...
  <br />  [19/20][1550/1583]	Loss_D: 0.0771	Loss_G: 4.4849	D(x): 0.9462	D(G(z)): 0.0188 / 0.0183
</summary>
<div markdown="1">
    Starting Training Loop...
  <br />  [0/20][0/1583]	Loss_D: 1.7410	Loss_G: 4.7765	D(x): 0.5343	D(G(z)): 0.5771 / 0.0136
  <br />  [0/20][50/1583]	Loss_D: 0.0449	Loss_G: 19.0586	D(x): 0.9719	D(G(z)): 0.0000 / 0.0000
  <br />  [0/20][100/1583]	Loss_D: 0.0672	Loss_G: 8.9777	D(x): 0.9594	D(G(z)): 0.0147 / 0.0004
  <br />  [0/20][150/1583]	Loss_D: 0.5382	Loss_G: 4.3727	D(x): 0.7767	D(G(z)): 0.1481 / 0.0210
  <br />  [0/20][200/1583]	Loss_D: 0.3375	Loss_G: 3.3296	D(x): 0.8734	D(G(z)): 0.1447 / 0.0512
  <br />  [0/20][250/1583]	Loss_D: 0.4448	Loss_G: 4.5643	D(x): 0.8256	D(G(z)): 0.1510 / 0.0214
  <br />  [0/20][300/1583]	Loss_D: 0.4503	Loss_G: 4.5324	D(x): 0.8277	D(G(z)): 0.1886 / 0.0190
  <br />  [0/20][350/1583]	Loss_D: 0.8060	Loss_G: 6.4533	D(x): 0.8264	D(G(z)): 0.3890 / 0.0032
  <br />  [0/20][400/1583]	Loss_D: 0.3180	Loss_G: 6.1294	D(x): 0.8139	D(G(z)): 0.0258 / 0.0059
  <br />  [0/20][450/1583]	Loss_D: 0.3987	Loss_G: 4.7085	D(x): 0.9028	D(G(z)): 0.2162 / 0.0174
  <br />  [0/20][500/1583]	Loss_D: 1.7161	Loss_G: 3.4000	D(x): 0.3963	D(G(z)): 0.0449 / 0.0692
  <br />  [0/20][550/1583]	Loss_D: 0.4825	Loss_G: 3.6144	D(x): 0.8382	D(G(z)): 0.2082 / 0.0410
  <br />  [0/20][600/1583]	Loss_D: 0.2545	Loss_G: 5.3959	D(x): 0.8620	D(G(z)): 0.0592 / 0.0087
  <br />  [0/20][650/1583]	Loss_D: 0.3354	Loss_G: 4.8481	D(x): 0.9222	D(G(z)): 0.1839 / 0.0137
  <br />  [0/20][700/1583]	Loss_D: 0.5186	Loss_G: 4.2392	D(x): 0.8203	D(G(z)): 0.2167 / 0.0238
  <br />  [0/20][750/1583]	Loss_D: 0.5599	Loss_G: 6.3192	D(x): 0.8593	D(G(z)): 0.2695 / 0.0034
  <br />  [0/20][800/1583]	Loss_D: 0.4188	Loss_G: 4.9431	D(x): 0.8882	D(G(z)): 0.2113 / 0.0144
  <br />  [0/20][850/1583]	Loss_D: 0.6197	Loss_G: 4.4217	D(x): 0.8696	D(G(z)): 0.3028 / 0.0209
  <br />  [0/20][900/1583]	Loss_D: 0.4814	Loss_G: 5.1496	D(x): 0.9114	D(G(z)): 0.2720 / 0.0113
  <br />  [0/20][950/1583]	Loss_D: 0.6159	Loss_G: 5.2550	D(x): 0.8648	D(G(z)): 0.2966 / 0.0095
  <br />  [0/20][1000/1583]	Loss_D: 2.1783	Loss_G: 4.7717	D(x): 0.2019	D(G(z)): 0.0007 / 0.0210
  <br />  [0/20][1050/1583]	Loss_D: 0.7515	Loss_G: 5.6444	D(x): 0.8597	D(G(z)): 0.3802 / 0.0078
  <br />  [0/20][1100/1583]	Loss_D: 1.5281	Loss_G: 9.7716	D(x): 0.9456	D(G(z)): 0.6747 / 0.0003
  <br />  [0/20][1150/1583]	Loss_D: 0.8123	Loss_G: 7.3049	D(x): 0.8826	D(G(z)): 0.4314 / 0.0018
  <br />  [0/20][1200/1583]	Loss_D: 0.8014	Loss_G: 1.9396	D(x): 0.5848	D(G(z)): 0.0779 / 0.1988
  <br />  [0/20][1250/1583]	Loss_D: 0.8756	Loss_G: 5.9078	D(x): 0.9042	D(G(z)): 0.4615 / 0.0059
  <br />  [0/20][1300/1583]	Loss_D: 0.5593	Loss_G: 2.9927	D(x): 0.6873	D(G(z)): 0.0803 / 0.0852
  <br />  [0/20][1350/1583]	Loss_D: 0.3988	Loss_G: 3.8558	D(x): 0.7763	D(G(z)): 0.0747 / 0.0365
  <br />  [0/20][1400/1583]	Loss_D: 0.3396	Loss_G: 4.0001	D(x): 0.8580	D(G(z)): 0.1349 / 0.0295
  <br />  [0/20][1450/1583]	Loss_D: 2.3297	Loss_G: 5.9231	D(x): 0.9679	D(G(z)): 0.7938 / 0.0094
  <br />  [0/20][1500/1583]	Loss_D: 0.2046	Loss_G: 4.1294	D(x): 0.8830	D(G(z)): 0.0545 / 0.0258
  <br />  [0/20][1550/1583]	Loss_D: 0.3662	Loss_G: 5.0539	D(x): 0.9373	D(G(z)): 0.2283 / 0.0126
  <br />  [1/20][0/1583]	Loss_D: 0.3495	Loss_G: 3.0891	D(x): 0.8007	D(G(z)): 0.0768 / 0.0711
  <br />  [1/20][50/1583]	Loss_D: 0.4281	Loss_G: 4.7255	D(x): 0.9287	D(G(z)): 0.2718 / 0.0132
  <br />  [1/20][100/1583]	Loss_D: 0.6284	Loss_G: 1.7680	D(x): 0.6295	D(G(z)): 0.0378 / 0.2508
  <br />  [1/20][150/1583]	Loss_D: 0.4197	Loss_G: 3.1495	D(x): 0.7910	D(G(z)): 0.1052 / 0.0629
  <br />  [1/20][200/1583]	Loss_D: 0.7653	Loss_G: 4.9047	D(x): 0.8667	D(G(z)): 0.3928 / 0.0137
  <br />  [1/20][250/1583]	Loss_D: 0.9562	Loss_G: 2.5036	D(x): 0.5037	D(G(z)): 0.0166 / 0.1667
  <br />  [1/20][300/1583]	Loss_D: 0.4809	Loss_G: 3.3706	D(x): 0.8228	D(G(z)): 0.1935 / 0.0586
  <br />  [1/20][350/1583]	Loss_D: 0.6087	Loss_G: 1.8009	D(x): 0.6726	D(G(z)): 0.1011 / 0.2359
  <br />  [1/20][400/1583]	Loss_D: 0.4401	Loss_G: 3.6131	D(x): 0.8195	D(G(z)): 0.1787 / 0.0393
  <br />  [1/20][450/1583]	Loss_D: 0.5013	Loss_G: 2.5153	D(x): 0.7831	D(G(z)): 0.1805 / 0.1068
  <br />  [1/20][500/1583]	Loss_D: 0.8559	Loss_G: 2.9979	D(x): 0.6304	D(G(z)): 0.1721 / 0.0934
  <br />  [1/20][550/1583]	Loss_D: 0.4231	Loss_G: 3.3755	D(x): 0.8317	D(G(z)): 0.1810 / 0.0516
  <br />  [1/20][600/1583]	Loss_D: 0.3380	Loss_G: 3.9220	D(x): 0.8639	D(G(z)): 0.1467 / 0.0317
  <br />  [1/20][650/1583]	Loss_D: 0.5970	Loss_G: 2.8443	D(x): 0.6796	D(G(z)): 0.1008 / 0.0839
  <br />  [1/20][700/1583]	Loss_D: 0.5490	Loss_G: 4.4962	D(x): 0.9450	D(G(z)): 0.3408 / 0.0224
  <br />  [1/20][750/1583]	Loss_D: 0.7478	Loss_G: 4.6962	D(x): 0.9188	D(G(z)): 0.4250 / 0.0150
  <br />  [1/20][800/1583]	Loss_D: 0.9041	Loss_G: 4.8681	D(x): 0.9307	D(G(z)): 0.4542 / 0.0205
  <br />  [1/20][850/1583]	Loss_D: 0.5160	Loss_G: 2.6778	D(x): 0.7839	D(G(z)): 0.1914 / 0.0959
  <br />  [1/20][900/1583]	Loss_D: 0.4130	Loss_G: 2.6591	D(x): 0.7604	D(G(z)): 0.0793 / 0.1079
  <br />  [1/20][950/1583]	Loss_D: 0.7728	Loss_G: 4.5145	D(x): 0.8560	D(G(z)): 0.3996 / 0.0224
  <br />  [1/20][1000/1583]	Loss_D: 0.6709	Loss_G: 2.4271	D(x): 0.6159	D(G(z)): 0.0616 / 0.1313
  <br />  [1/20][1050/1583]	Loss_D: 0.6210	Loss_G: 1.9101	D(x): 0.6557	D(G(z)): 0.1032 / 0.1963
  <br />  [1/20][1100/1583]	Loss_D: 0.6920	Loss_G: 4.6043	D(x): 0.8611	D(G(z)): 0.3601 / 0.0176
  <br />  [1/20][1150/1583]	Loss_D: 0.6916	Loss_G: 1.8056	D(x): 0.6420	D(G(z)): 0.1577 / 0.2154
  <br />  [1/20][1200/1583]	Loss_D: 0.5857	Loss_G: 3.0779	D(x): 0.9126	D(G(z)): 0.3339 / 0.0785
  <br />  [1/20][1250/1583]	Loss_D: 0.8223	Loss_G: 1.8331	D(x): 0.5239	D(G(z)): 0.0438 / 0.2263
  <br />  [1/20][1300/1583]	Loss_D: 2.0921	Loss_G: 7.0854	D(x): 0.9861	D(G(z)): 0.8167 / 0.0015
  <br />  [1/20][1350/1583]	Loss_D: 0.5447	Loss_G: 2.3173	D(x): 0.7137	D(G(z)): 0.1482 / 0.1267
  <br />  [1/20][1400/1583]	Loss_D: 1.5583	Loss_G: 7.0886	D(x): 0.9757	D(G(z)): 0.7073 / 0.0018
  <br />  [1/20][1450/1583]	Loss_D: 0.5013	Loss_G: 3.0252	D(x): 0.8100	D(G(z)): 0.2229 / 0.0701
  <br />  [1/20][1500/1583]	Loss_D: 1.2264	Loss_G: 3.5587	D(x): 0.8921	D(G(z)): 0.6020 / 0.0627
  <br />  [1/20][1550/1583]	Loss_D: 0.5227	Loss_G: 2.5838	D(x): 0.7990	D(G(z)): 0.2174 / 0.0988
  <br />  [2/20][0/1583]	Loss_D: 0.4165	Loss_G: 2.4796	D(x): 0.7733	D(G(z)): 0.1202 / 0.1085
  <br />  [2/20][50/1583]	Loss_D: 0.8466	Loss_G: 1.9751	D(x): 0.5038	D(G(z)): 0.0321 / 0.1874
  <br />  [2/20][100/1583]	Loss_D: 0.4960	Loss_G: 2.0882	D(x): 0.6730	D(G(z)): 0.0633 / 0.1603
  <br />  [2/20][150/1583]	Loss_D: 0.7157	Loss_G: 1.6555	D(x): 0.7571	D(G(z)): 0.2876 / 0.2416
  <br />  [2/20][200/1583]	Loss_D: 0.5810	Loss_G: 1.8568	D(x): 0.6774	D(G(z)): 0.1262 / 0.1943
  <br />  [2/20][250/1583]	Loss_D: 0.8586	Loss_G: 1.9327	D(x): 0.5397	D(G(z)): 0.1100 / 0.2021
  <br />  [2/20][300/1583]	Loss_D: 1.1256	Loss_G: 4.0812	D(x): 0.9422	D(G(z)): 0.5831 / 0.0394
  <br />  [2/20][350/1583]	Loss_D: 1.1133	Loss_G: 0.6896	D(x): 0.4193	D(G(z)): 0.0297 / 0.5650
  <br />  [2/20][400/1583]	Loss_D: 0.7344	Loss_G: 2.5044	D(x): 0.5987	D(G(z)): 0.0796 / 0.1437
  <br />  [2/20][450/1583]	Loss_D: 0.8426	Loss_G: 3.7049	D(x): 0.9143	D(G(z)): 0.4675 / 0.0368
  <br />  [2/20][500/1583]	Loss_D: 0.8461	Loss_G: 4.4628	D(x): 0.9014	D(G(z)): 0.4809 / 0.0165
  <br />  [2/20][550/1583]	Loss_D: 0.7044	Loss_G: 0.9616	D(x): 0.5811	D(G(z)): 0.0680 / 0.4335
  <br />  [2/20][600/1583]	Loss_D: 0.4981	Loss_G: 2.5887	D(x): 0.7610	D(G(z)): 0.1710 / 0.0997
  <br />  [2/20][650/1583]	Loss_D: 0.5940	Loss_G: 2.0112	D(x): 0.6949	D(G(z)): 0.1572 / 0.1693
  <br />  [2/20][700/1583]	Loss_D: 1.0516	Loss_G: 0.9519	D(x): 0.4300	D(G(z)): 0.0271 / 0.4620
  <br />  [2/20][750/1583]	Loss_D: 0.6836	Loss_G: 3.1258	D(x): 0.8643	D(G(z)): 0.3775 / 0.0577
  <br />  [2/20][800/1583]	Loss_D: 0.7599	Loss_G: 3.7076	D(x): 0.9092	D(G(z)): 0.4526 / 0.0341
  <br />  [2/20][850/1583]	Loss_D: 0.5043	Loss_G: 3.1779	D(x): 0.8870	D(G(z)): 0.2876 / 0.0588
  <br />  [2/20][900/1583]	Loss_D: 0.7738	Loss_G: 3.0952	D(x): 0.8357	D(G(z)): 0.4029 / 0.0594
  <br />  [2/20][950/1583]	Loss_D: 0.4681	Loss_G: 2.7019	D(x): 0.7963	D(G(z)): 0.1882 / 0.0807
  <br />  [2/20][1000/1583]	Loss_D: 0.6668	Loss_G: 2.4374	D(x): 0.7022	D(G(z)): 0.2223 / 0.1057
  <br />  [2/20][1050/1583]	Loss_D: 0.5939	Loss_G: 2.1903	D(x): 0.7709	D(G(z)): 0.2344 / 0.1404
  <br />  [2/20][1100/1583]	Loss_D: 0.4734	Loss_G: 3.1788	D(x): 0.8460	D(G(z)): 0.2437 / 0.0518
  <br />  [2/20][1150/1583]	Loss_D: 0.6478	Loss_G: 3.2170	D(x): 0.8803	D(G(z)): 0.3519 / 0.0618
  <br />  [2/20][1200/1583]	Loss_D: 0.7725	Loss_G: 1.6971	D(x): 0.5410	D(G(z)): 0.0644 / 0.2365
  <br />  [2/20][1250/1583]	Loss_D: 0.8769	Loss_G: 3.8022	D(x): 0.9590	D(G(z)): 0.5094 / 0.0359
  <br />  [2/20][1300/1583]	Loss_D: 0.4804	Loss_G: 2.4144	D(x): 0.7237	D(G(z)): 0.1067 / 0.1161
  <br />  [2/20][1350/1583]	Loss_D: 0.7312	Loss_G: 3.4400	D(x): 0.8520	D(G(z)): 0.3902 / 0.0454
  <br />  [2/20][1400/1583]	Loss_D: 1.3515	Loss_G: 4.9463	D(x): 0.9638	D(G(z)): 0.6742 / 0.0119
  <br />  [2/20][1450/1583]	Loss_D: 0.8293	Loss_G: 1.2908	D(x): 0.5222	D(G(z)): 0.0623 / 0.3216
  <br />  [2/20][1500/1583]	Loss_D: 0.6314	Loss_G: 2.3523	D(x): 0.7369	D(G(z)): 0.2355 / 0.1251
  <br />  [2/20][1550/1583]	Loss_D: 1.2016	Loss_G: 4.7539	D(x): 0.9094	D(G(z)): 0.6105 / 0.0145
  <br />  [3/20][0/1583]	Loss_D: 0.7443	Loss_G: 4.0368	D(x): 0.8949	D(G(z)): 0.4305 / 0.0251
  <br />  [3/20][50/1583]	Loss_D: 0.5733	Loss_G: 2.9640	D(x): 0.8347	D(G(z)): 0.2895 / 0.0732
  <br />  [3/20][100/1583]	Loss_D: 0.6174	Loss_G: 2.7364	D(x): 0.7529	D(G(z)): 0.2465 / 0.0810
  <br />  [3/20][150/1583]	Loss_D: 1.1024	Loss_G: 4.4745	D(x): 0.8992	D(G(z)): 0.5760 / 0.0167
  <br />  [3/20][200/1583]	Loss_D: 0.4080	Loss_G: 2.4299	D(x): 0.7861	D(G(z)): 0.1361 / 0.1078
  <br />  [3/20][250/1583]	Loss_D: 0.7814	Loss_G: 1.3794	D(x): 0.5675	D(G(z)): 0.1168 / 0.3091
  <br />  [3/20][300/1583]	Loss_D: 1.5277	Loss_G: 1.0074	D(x): 0.2852	D(G(z)): 0.0417 / 0.4186
  <br />  [3/20][350/1583]	Loss_D: 0.5269	Loss_G: 2.1274	D(x): 0.7573	D(G(z)): 0.1816 / 0.1459
  <br />  [3/20][400/1583]	Loss_D: 0.5345	Loss_G: 2.4768	D(x): 0.7594	D(G(z)): 0.1945 / 0.1082
  <br />  [3/20][450/1583]	Loss_D: 0.6973	Loss_G: 2.7010	D(x): 0.7838	D(G(z)): 0.3241 / 0.0932
  <br />  [3/20][500/1583]	Loss_D: 0.9344	Loss_G: 3.8631	D(x): 0.9555	D(G(z)): 0.5286 / 0.0309
  <br />  [3/20][550/1583]	Loss_D: 0.7371	Loss_G: 3.9111	D(x): 0.9194	D(G(z)): 0.4313 / 0.0284
  <br />  [3/20][600/1583]	Loss_D: 0.7451	Loss_G: 3.1238	D(x): 0.8919	D(G(z)): 0.4267 / 0.0583
  <br />  [3/20][650/1583]	Loss_D: 0.7557	Loss_G: 3.9020	D(x): 0.8880	D(G(z)): 0.4247 / 0.0348
  <br />  [3/20][700/1583]	Loss_D: 0.5578	Loss_G: 2.9881	D(x): 0.8846	D(G(z)): 0.3224 / 0.0665
  <br />  [3/20][750/1583]	Loss_D: 0.6100	Loss_G: 3.1242	D(x): 0.8720	D(G(z)): 0.3423 / 0.0601
  <br />  [3/20][800/1583]	Loss_D: 0.5605	Loss_G: 2.7928	D(x): 0.7818	D(G(z)): 0.2393 / 0.0797
  <br />  [3/20][850/1583]	Loss_D: 0.6751	Loss_G: 1.9463	D(x): 0.6537	D(G(z)): 0.1695 / 0.1814
  <br />  [3/20][900/1583]	Loss_D: 1.0268	Loss_G: 0.7435	D(x): 0.4494	D(G(z)): 0.0827 / 0.5238
  <br />  [3/20][950/1583]	Loss_D: 0.9942	Loss_G: 3.3982	D(x): 0.9163	D(G(z)): 0.5413 / 0.0463
  <br />  [3/20][1000/1583]	Loss_D: 0.6281	Loss_G: 1.4484	D(x): 0.6802	D(G(z)): 0.1766 / 0.2706
  <br />  [3/20][1050/1583]	Loss_D: 1.5523	Loss_G: 4.4541	D(x): 0.9091	D(G(z)): 0.6790 / 0.0222
  <br />  [3/20][1100/1583]	Loss_D: 0.5266	Loss_G: 1.7666	D(x): 0.7645	D(G(z)): 0.1886 / 0.2058
  <br />  [3/20][1150/1583]	Loss_D: 0.6640	Loss_G: 3.9673	D(x): 0.8396	D(G(z)): 0.3543 / 0.0261
  <br />  [3/20][1200/1583]	Loss_D: 0.5495	Loss_G: 2.7730	D(x): 0.6856	D(G(z)): 0.1013 / 0.0908
  <br />  [3/20][1250/1583]	Loss_D: 0.7748	Loss_G: 3.4206	D(x): 0.8361	D(G(z)): 0.4071 / 0.0443
  <br />  [3/20][1300/1583]	Loss_D: 1.0531	Loss_G: 0.7425	D(x): 0.4492	D(G(z)): 0.0930 / 0.5158
  <br />  [3/20][1350/1583]	Loss_D: 1.2218	Loss_G: 3.6362	D(x): 0.8856	D(G(z)): 0.5945 / 0.0399
  <br />  [3/20][1400/1583]	Loss_D: 0.5066	Loss_G: 2.4130	D(x): 0.7957	D(G(z)): 0.2118 / 0.1101
  <br />  [3/20][1450/1583]	Loss_D: 1.0593	Loss_G: 1.1941	D(x): 0.4555	D(G(z)): 0.1129 / 0.3577
  <br />  [3/20][1500/1583]	Loss_D: 0.8114	Loss_G: 3.4493	D(x): 0.8202	D(G(z)): 0.4114 / 0.0421
  <br />  [3/20][1550/1583]	Loss_D: 0.6355	Loss_G: 2.7985	D(x): 0.7973	D(G(z)): 0.3010 / 0.0773
  <br />  [4/20][0/1583]	Loss_D: 0.6190	Loss_G: 1.4828	D(x): 0.6865	D(G(z)): 0.1701 / 0.2686
  <br />  [4/20][50/1583]	Loss_D: 0.7670	Loss_G: 1.5755	D(x): 0.5785	D(G(z)): 0.1064 / 0.2610
  <br />  [4/20][100/1583]	Loss_D: 0.9794	Loss_G: 0.8689	D(x): 0.4682	D(G(z)): 0.0713 / 0.4803
  <br />  [4/20][150/1583]	Loss_D: 0.4796	Loss_G: 2.5772	D(x): 0.7851	D(G(z)): 0.1844 / 0.0981
  <br />  [4/20][200/1583]	Loss_D: 1.0336	Loss_G: 4.0929	D(x): 0.9576	D(G(z)): 0.5638 / 0.0253
  <br />  [4/20][250/1583]	Loss_D: 0.7909	Loss_G: 4.3225	D(x): 0.9198	D(G(z)): 0.4664 / 0.0217
  <br />  [4/20][300/1583]	Loss_D: 2.3641	Loss_G: 5.2525	D(x): 0.9666	D(G(z)): 0.8407 / 0.0107
  <br />  [4/20][350/1583]	Loss_D: 0.3944	Loss_G: 2.9269	D(x): 0.8325	D(G(z)): 0.1671 / 0.0670
  <br />  [4/20][400/1583]	Loss_D: 0.6390	Loss_G: 2.0675	D(x): 0.6757	D(G(z)): 0.1704 / 0.1584
  <br />  [4/20][450/1583]	Loss_D: 0.7568	Loss_G: 3.3020	D(x): 0.8837	D(G(z)): 0.4247 / 0.0513
  <br />  [4/20][500/1583]	Loss_D: 0.5173	Loss_G: 4.1190	D(x): 0.9339	D(G(z)): 0.3338 / 0.0214
  <br />  [4/20][550/1583]	Loss_D: 0.7709	Loss_G: 2.0782	D(x): 0.8477	D(G(z)): 0.3988 / 0.1619
  <br />  [4/20][600/1583]	Loss_D: 0.6134	Loss_G: 2.2465	D(x): 0.6198	D(G(z)): 0.0796 / 0.1321
  <br />  [4/20][650/1583]	Loss_D: 1.1340	Loss_G: 4.1819	D(x): 0.9502	D(G(z)): 0.6112 / 0.0210
  <br />  [4/20][700/1583]	Loss_D: 0.3680	Loss_G: 2.5364	D(x): 0.8330	D(G(z)): 0.1506 / 0.1024
  <br />  [4/20][750/1583]	Loss_D: 0.7582	Loss_G: 3.5181	D(x): 0.8960	D(G(z)): 0.4361 / 0.0377
  <br />  [4/20][800/1583]	Loss_D: 0.5314	Loss_G: 2.8887	D(x): 0.8143	D(G(z)): 0.2457 / 0.0748
  <br />  [4/20][850/1583]	Loss_D: 0.7218	Loss_G: 3.3921	D(x): 0.8423	D(G(z)): 0.3714 / 0.0483
  <br />  [4/20][900/1583]	Loss_D: 0.7013	Loss_G: 2.3325	D(x): 0.6629	D(G(z)): 0.1797 / 0.1271
  <br />  [4/20][950/1583]	Loss_D: 0.4679	Loss_G: 2.7841	D(x): 0.8424	D(G(z)): 0.2300 / 0.0829
  <br />  [4/20][1000/1583]	Loss_D: 0.4535	Loss_G: 2.3732	D(x): 0.8321	D(G(z)): 0.2089 / 0.1128
  <br />  [4/20][1050/1583]	Loss_D: 0.6944	Loss_G: 3.3689	D(x): 0.8595	D(G(z)): 0.3831 / 0.0454
  <br />  [4/20][1100/1583]	Loss_D: 0.9457	Loss_G: 0.9144	D(x): 0.4850	D(G(z)): 0.0754 / 0.4504
  <br />  [4/20][1150/1583]	Loss_D: 0.6403	Loss_G: 3.7571	D(x): 0.8796	D(G(z)): 0.3643 / 0.0326
  <br />  [4/20][1200/1583]	Loss_D: 1.6224	Loss_G: 0.5884	D(x): 0.2613	D(G(z)): 0.0330 / 0.6027
  <br />  [4/20][1250/1583]	Loss_D: 0.8926	Loss_G: 2.1653	D(x): 0.6066	D(G(z)): 0.2507 / 0.1563
  <br />  [4/20][1300/1583]	Loss_D: 0.5672	Loss_G: 1.8014	D(x): 0.6662	D(G(z)): 0.1085 / 0.2015
  <br />  [4/20][1350/1583]	Loss_D: 0.6621	Loss_G: 4.0858	D(x): 0.9298	D(G(z)): 0.4096 / 0.0223
  <br />  [4/20][1400/1583]	Loss_D: 0.6584	Loss_G: 3.2440	D(x): 0.8878	D(G(z)): 0.3851 / 0.0501
  <br />  [4/20][1450/1583]	Loss_D: 0.6099	Loss_G: 3.2473	D(x): 0.9127	D(G(z)): 0.3747 / 0.0492
  <br />  [4/20][1500/1583]	Loss_D: 1.6107	Loss_G: 0.2481	D(x): 0.2599	D(G(z)): 0.0178 / 0.8148
  <br />  [4/20][1550/1583]	Loss_D: 0.4393	Loss_G: 2.5206	D(x): 0.7341	D(G(z)): 0.0924 / 0.1040
  <br />  [5/20][0/1583]	Loss_D: 0.7323	Loss_G: 3.4426	D(x): 0.8958	D(G(z)): 0.4190 / 0.0455
  <br />  [5/20][50/1583]	Loss_D: 0.4701	Loss_G: 2.9693	D(x): 0.8395	D(G(z)): 0.2326 / 0.0673
  <br />  [5/20][100/1583]	Loss_D: 0.6914	Loss_G: 3.9368	D(x): 0.9114	D(G(z)): 0.4031 / 0.0270
  <br />  [5/20][150/1583]	Loss_D: 0.5457	Loss_G: 2.3770	D(x): 0.7511	D(G(z)): 0.1965 / 0.1219
  <br />  [5/20][200/1583]	Loss_D: 0.5658	Loss_G: 1.5108	D(x): 0.6612	D(G(z)): 0.0848 / 0.2547
  <br />  [5/20][250/1583]	Loss_D: 0.6146	Loss_G: 3.9287	D(x): 0.9534	D(G(z)): 0.4005 / 0.0259
  <br />  [5/20][300/1583]	Loss_D: 0.9300	Loss_G: 3.3197	D(x): 0.8995	D(G(z)): 0.5042 / 0.0502
  <br />  [5/20][350/1583]	Loss_D: 0.5325	Loss_G: 1.6093	D(x): 0.7229	D(G(z)): 0.1440 / 0.2417
  <br />  [5/20][400/1583]	Loss_D: 0.9521	Loss_G: 0.4797	D(x): 0.4600	D(G(z)): 0.0518 / 0.6548
  <br />  [5/20][450/1583]	Loss_D: 0.7482	Loss_G: 3.2963	D(x): 0.9374	D(G(z)): 0.4515 / 0.0494
  <br />  [5/20][500/1583]	Loss_D: 1.2473	Loss_G: 4.8589	D(x): 0.9432	D(G(z)): 0.6380 / 0.0113
  <br />  [5/20][550/1583]	Loss_D: 0.5828	Loss_G: 1.3968	D(x): 0.6884	D(G(z)): 0.1459 / 0.2952
  <br />  [5/20][600/1583]	Loss_D: 0.4951	Loss_G: 3.5542	D(x): 0.9001	D(G(z)): 0.2957 / 0.0384
  <br />  [5/20][650/1583]	Loss_D: 0.4813	Loss_G: 3.2853	D(x): 0.8736	D(G(z)): 0.2686 / 0.0486
  <br />  [5/20][700/1583]	Loss_D: 0.5033	Loss_G: 2.6258	D(x): 0.8170	D(G(z)): 0.2208 / 0.0961
  <br />  [5/20][750/1583]	Loss_D: 0.6739	Loss_G: 1.9801	D(x): 0.6349	D(G(z)): 0.1067 / 0.1897
  <br />  [5/20][800/1583]	Loss_D: 0.4293	Loss_G: 2.6046	D(x): 0.8100	D(G(z)): 0.1678 / 0.0933
  <br />  [5/20][850/1583]	Loss_D: 0.5649	Loss_G: 2.2224	D(x): 0.6974	D(G(z)): 0.1357 / 0.1301
  <br />  [5/20][900/1583]	Loss_D: 1.1035	Loss_G: 0.8224	D(x): 0.4289	D(G(z)): 0.0525 / 0.4958
  <br />  [5/20][950/1583]	Loss_D: 3.2330	Loss_G: 0.0733	D(x): 0.0794	D(G(z)): 0.0310 / 0.9343
  <br />  [5/20][1000/1583]	Loss_D: 0.6679	Loss_G: 1.5752	D(x): 0.6013	D(G(z)): 0.0840 / 0.2559
  <br />  [5/20][1050/1583]	Loss_D: 1.1246	Loss_G: 4.7649	D(x): 0.9467	D(G(z)): 0.6041 / 0.0127
  <br />  [5/20][1100/1583]	Loss_D: 0.8038	Loss_G: 1.6691	D(x): 0.6885	D(G(z)): 0.2793 / 0.2359
  <br />  [5/20][1150/1583]	Loss_D: 0.6897	Loss_G: 4.0272	D(x): 0.8872	D(G(z)): 0.3930 / 0.0270
  <br />  [5/20][1200/1583]	Loss_D: 1.8038	Loss_G: 6.2461	D(x): 0.9885	D(G(z)): 0.7836 / 0.0040
  <br />  [5/20][1250/1583]	Loss_D: 0.9141	Loss_G: 1.4521	D(x): 0.4791	D(G(z)): 0.0559 / 0.2952
  <br />  [5/20][1300/1583]	Loss_D: 0.5217	Loss_G: 2.2163	D(x): 0.7537	D(G(z)): 0.1791 / 0.1389
  <br />  [5/20][1350/1583]	Loss_D: 0.4390	Loss_G: 2.2286	D(x): 0.8911	D(G(z)): 0.2451 / 0.1387
  <br />  [5/20][1400/1583]	Loss_D: 0.6742	Loss_G: 1.4092	D(x): 0.6256	D(G(z)): 0.1300 / 0.2963
  <br />  [5/20][1450/1583]	Loss_D: 0.3419	Loss_G: 2.7905	D(x): 0.8096	D(G(z)): 0.0997 / 0.0881
  <br />  [5/20][1500/1583]	Loss_D: 0.4997	Loss_G: 2.9426	D(x): 0.8166	D(G(z)): 0.2252 / 0.0671
  <br />  [5/20][1550/1583]	Loss_D: 0.3468	Loss_G: 2.8488	D(x): 0.9008	D(G(z)): 0.1978 / 0.0749
  <br />  [6/20][0/1583]	Loss_D: 4.3955	Loss_G: 5.5303	D(x): 0.9860	D(G(z)): 0.9622 / 0.0088
  <br />  [6/20][50/1583]	Loss_D: 0.7677	Loss_G: 1.8061	D(x): 0.5506	D(G(z)): 0.0690 / 0.2167
  <br />  [6/20][100/1583]	Loss_D: 0.7837	Loss_G: 3.6642	D(x): 0.8930	D(G(z)): 0.4401 / 0.0357
  <br />  [6/20][150/1583]	Loss_D: 0.3844	Loss_G: 3.2250	D(x): 0.8740	D(G(z)): 0.2014 / 0.0530
  <br />  [6/20][200/1583]	Loss_D: 0.3511	Loss_G: 3.5891	D(x): 0.8962	D(G(z)): 0.1963 / 0.0378
  <br />  [6/20][250/1583]	Loss_D: 0.5464	Loss_G: 1.9196	D(x): 0.6527	D(G(z)): 0.0583 / 0.1851
  <br />  [6/20][300/1583]	Loss_D: 0.6648	Loss_G: 2.0872	D(x): 0.6812	D(G(z)): 0.1896 / 0.1551
  <br />  [6/20][350/1583]	Loss_D: 0.5900	Loss_G: 4.0803	D(x): 0.9030	D(G(z)): 0.3360 / 0.0253
  <br />  [6/20][400/1583]	Loss_D: 0.4268	Loss_G: 2.8039	D(x): 0.8247	D(G(z)): 0.1793 / 0.0828
  <br />  [6/20][450/1583]	Loss_D: 0.4140	Loss_G: 2.9813	D(x): 0.9032	D(G(z)): 0.2454 / 0.0668
  <br />  [6/20][500/1583]	Loss_D: 0.3369	Loss_G: 2.3742	D(x): 0.8204	D(G(z)): 0.1106 / 0.1231
  <br />  [6/20][550/1583]	Loss_D: 1.0842	Loss_G: 0.3029	D(x): 0.4080	D(G(z)): 0.0300 / 0.7758
  <br />  [6/20][600/1583]	Loss_D: 0.6743	Loss_G: 2.1038	D(x): 0.5983	D(G(z)): 0.0673 / 0.1638
  <br />  [6/20][650/1583]	Loss_D: 0.5764	Loss_G: 3.9941	D(x): 0.9302	D(G(z)): 0.3584 / 0.0268
  <br />  [6/20][700/1583]	Loss_D: 0.3742	Loss_G: 3.2708	D(x): 0.9046	D(G(z)): 0.2213 / 0.0492
  <br />  [6/20][750/1583]	Loss_D: 1.1182	Loss_G: 1.0595	D(x): 0.4141	D(G(z)): 0.0312 / 0.4134
  <br />  [6/20][800/1583]	Loss_D: 0.5230	Loss_G: 2.1190	D(x): 0.7189	D(G(z)): 0.1305 / 0.1560
  <br />  [6/20][850/1583]	Loss_D: 1.0702	Loss_G: 1.6753	D(x): 0.6622	D(G(z)): 0.3463 / 0.3068
  <br />  [6/20][900/1583]	Loss_D: 0.5835	Loss_G: 1.5507	D(x): 0.6592	D(G(z)): 0.0960 / 0.2584
  <br />  [6/20][950/1583]	Loss_D: 1.3041	Loss_G: 1.0619	D(x): 0.3352	D(G(z)): 0.0179 / 0.4241
  <br />  [6/20][1000/1583]	Loss_D: 0.7321	Loss_G: 1.1229	D(x): 0.5704	D(G(z)): 0.0844 / 0.3779
  <br />  [6/20][1050/1583]	Loss_D: 0.5501	Loss_G: 2.7354	D(x): 0.8075	D(G(z)): 0.2538 / 0.0825
  <br />  [6/20][1100/1583]	Loss_D: 0.4789	Loss_G: 2.2263	D(x): 0.7961	D(G(z)): 0.1873 / 0.1340
  <br />  [6/20][1150/1583]	Loss_D: 0.6680	Loss_G: 4.6801	D(x): 0.9348	D(G(z)): 0.4143 / 0.0146
  <br />  [6/20][1200/1583]	Loss_D: 0.4431	Loss_G: 3.3127	D(x): 0.8719	D(G(z)): 0.2343 / 0.0534
  <br />  [6/20][1250/1583]	Loss_D: 0.6206	Loss_G: 4.6355	D(x): 0.9396	D(G(z)): 0.3845 / 0.0136
  <br />  [6/20][1300/1583]	Loss_D: 0.4057	Loss_G: 3.4378	D(x): 0.9003	D(G(z)): 0.2347 / 0.0449
  <br />  [6/20][1350/1583]	Loss_D: 0.6685	Loss_G: 1.7244	D(x): 0.6786	D(G(z)): 0.1996 / 0.2262
  <br />  [6/20][1400/1583]	Loss_D: 0.5023	Loss_G: 3.1058	D(x): 0.8602	D(G(z)): 0.2637 / 0.0586
  <br />  [6/20][1450/1583]	Loss_D: 0.6067	Loss_G: 2.4539	D(x): 0.7353	D(G(z)): 0.2084 / 0.1132
  <br />  [6/20][1500/1583]	Loss_D: 0.5930	Loss_G: 1.9310	D(x): 0.7278	D(G(z)): 0.1973 / 0.1740
  <br />  [6/20][1550/1583]	Loss_D: 0.3919	Loss_G: 2.8650	D(x): 0.8726	D(G(z)): 0.2034 / 0.0720
  <br />  [7/20][0/1583]	Loss_D: 0.9937	Loss_G: 0.8772	D(x): 0.4278	D(G(z)): 0.0203 / 0.4746
  <br />  [7/20][50/1583]	Loss_D: 0.5583	Loss_G: 2.2141	D(x): 0.6588	D(G(z)): 0.0663 / 0.1556
  <br />  [7/20][100/1583]	Loss_D: 0.4775	Loss_G: 2.8164	D(x): 0.7293	D(G(z)): 0.1121 / 0.0904
  <br />  [7/20][150/1583]	Loss_D: 0.3136	Loss_G: 3.4618	D(x): 0.8864	D(G(z)): 0.1531 / 0.0468
  <br />  [7/20][200/1583]	Loss_D: 0.3653	Loss_G: 3.7723	D(x): 0.9626	D(G(z)): 0.2593 / 0.0319
  <br />  [7/20][250/1583]	Loss_D: 0.3479	Loss_G: 3.3227	D(x): 0.8569	D(G(z)): 0.1571 / 0.0502
  <br />  [7/20][300/1583]	Loss_D: 0.7672	Loss_G: 5.0311	D(x): 0.9460	D(G(z)): 0.4491 / 0.0096
  <br />  [7/20][350/1583]	Loss_D: 0.5022	Loss_G: 2.7576	D(x): 0.8297	D(G(z)): 0.2316 / 0.0908
  <br />  [7/20][400/1583]	Loss_D: 0.4674	Loss_G: 1.9166	D(x): 0.7104	D(G(z)): 0.0751 / 0.1838
  <br />  [7/20][450/1583]	Loss_D: 0.4497	Loss_G: 3.4482	D(x): 0.8363	D(G(z)): 0.2027 / 0.0484
  <br />  [7/20][500/1583]	Loss_D: 0.4457	Loss_G: 2.0742	D(x): 0.7356	D(G(z)): 0.0974 / 0.1614
  <br />  [7/20][550/1583]	Loss_D: 0.3718	Loss_G: 2.7557	D(x): 0.8532	D(G(z)): 0.1646 / 0.0854
  <br />  [7/20][600/1583]	Loss_D: 0.3986	Loss_G: 3.1520	D(x): 0.9168	D(G(z)): 0.2433 / 0.0574
  <br />  [7/20][650/1583]	Loss_D: 0.4891	Loss_G: 3.9681	D(x): 0.9336	D(G(z)): 0.3129 / 0.0259
  <br />  [7/20][700/1583]	Loss_D: 0.8796	Loss_G: 4.7294	D(x): 0.9223	D(G(z)): 0.4910 / 0.0140
  <br />  [7/20][750/1583]	Loss_D: 1.3281	Loss_G: 0.7828	D(x): 0.3252	D(G(z)): 0.0342 / 0.5200
  <br />  [7/20][800/1583]	Loss_D: 0.3356	Loss_G: 3.2801	D(x): 0.9067	D(G(z)): 0.1936 / 0.0518
  <br />  [7/20][850/1583]	Loss_D: 0.6749	Loss_G: 4.3375	D(x): 0.9408	D(G(z)): 0.4078 / 0.0178
  <br />  [7/20][900/1583]	Loss_D: 0.6806	Loss_G: 1.4028	D(x): 0.6030	D(G(z)): 0.0648 / 0.3167
  <br />  [7/20][950/1583]	Loss_D: 0.5828	Loss_G: 3.9035	D(x): 0.8742	D(G(z)): 0.3122 / 0.0286
  <br />  [7/20][1000/1583]	Loss_D: 1.5289	Loss_G: 5.3634	D(x): 0.9872	D(G(z)): 0.7196 / 0.0085
  <br />  [7/20][1050/1583]	Loss_D: 0.3579	Loss_G: 3.0619	D(x): 0.8812	D(G(z)): 0.1914 / 0.0616
  <br />  [7/20][1100/1583]	Loss_D: 0.4081	Loss_G: 3.3457	D(x): 0.8707	D(G(z)): 0.2060 / 0.0460
  <br />  [7/20][1150/1583]	Loss_D: 0.3850	Loss_G: 1.8594	D(x): 0.7970	D(G(z)): 0.1164 / 0.1901
  <br />  [7/20][1200/1583]	Loss_D: 0.4083	Loss_G: 3.4403	D(x): 0.8714	D(G(z)): 0.2111 / 0.0443
  <br />  [7/20][1250/1583]	Loss_D: 0.4171	Loss_G: 3.0643	D(x): 0.7392	D(G(z)): 0.0596 / 0.0699
  <br />  [7/20][1300/1583]	Loss_D: 0.4492	Loss_G: 3.0841	D(x): 0.8419	D(G(z)): 0.2077 / 0.0663
  <br />  [7/20][1350/1583]	Loss_D: 0.3917	Loss_G: 3.9511	D(x): 0.9457	D(G(z)): 0.2551 / 0.0273
  <br />  [7/20][1400/1583]	Loss_D: 1.1657	Loss_G: 4.6778	D(x): 0.8444	D(G(z)): 0.5540 / 0.0164
  <br />  [7/20][1450/1583]	Loss_D: 0.7636	Loss_G: 4.7968	D(x): 0.9504	D(G(z)): 0.4558 / 0.0123
  <br />  [7/20][1500/1583]	Loss_D: 0.4004	Loss_G: 3.3350	D(x): 0.9098	D(G(z)): 0.2238 / 0.0464
  <br />  [7/20][1550/1583]	Loss_D: 1.4643	Loss_G: 1.1699	D(x): 0.2923	D(G(z)): 0.0120 / 0.3747
  <br />  [8/20][0/1583]	Loss_D: 0.3794	Loss_G: 2.8846	D(x): 0.8922	D(G(z)): 0.2056 / 0.0794
  <br />  [8/20][50/1583]	Loss_D: 0.3324	Loss_G: 3.7705	D(x): 0.9425	D(G(z)): 0.2165 / 0.0323
  <br />  [8/20][100/1583]	Loss_D: 1.4623	Loss_G: 5.1932	D(x): 0.9460	D(G(z)): 0.6943 / 0.0102
  <br />  [8/20][150/1583]	Loss_D: 0.2513	Loss_G: 2.8608	D(x): 0.8913	D(G(z)): 0.1138 / 0.0854
  <br />  [8/20][200/1583]	Loss_D: 0.4089	Loss_G: 1.9675	D(x): 0.7585	D(G(z)): 0.0976 / 0.1663
  <br />  [8/20][250/1583]	Loss_D: 0.5234	Loss_G: 4.9561	D(x): 0.9518	D(G(z)): 0.3438 / 0.0107
  <br />  [8/20][300/1583]	Loss_D: 0.4831	Loss_G: 3.2577	D(x): 0.8379	D(G(z)): 0.2316 / 0.0580
  <br />  [8/20][350/1583]	Loss_D: 0.6230	Loss_G: 1.5389	D(x): 0.6033	D(G(z)): 0.0506 / 0.2675
  <br />  [8/20][400/1583]	Loss_D: 0.4021	Loss_G: 1.8503	D(x): 0.7259	D(G(z)): 0.0341 / 0.2006
  <br />  [8/20][450/1583]	Loss_D: 0.3415	Loss_G: 3.3156	D(x): 0.9314	D(G(z)): 0.2180 / 0.0479
  <br />  [8/20][500/1583]	Loss_D: 0.7203	Loss_G: 1.0026	D(x): 0.5825	D(G(z)): 0.0615 / 0.4269
  <br />  [8/20][550/1583]	Loss_D: 0.4582	Loss_G: 1.6969	D(x): 0.7214	D(G(z)): 0.0910 / 0.2324
  <br />  [8/20][600/1583]	Loss_D: 1.4420	Loss_G: 1.1809	D(x): 0.3260	D(G(z)): 0.0394 / 0.3807
  <br />  [8/20][650/1583]	Loss_D: 0.2668	Loss_G: 3.3727	D(x): 0.9348	D(G(z)): 0.1705 / 0.0436
  <br />  [8/20][700/1583]	Loss_D: 0.7360	Loss_G: 1.0637	D(x): 0.5771	D(G(z)): 0.0725 / 0.3984
  <br />  [8/20][750/1583]	Loss_D: 0.4526	Loss_G: 3.5518	D(x): 0.8779	D(G(z)): 0.2426 / 0.0381
  <br />  [8/20][800/1583]	Loss_D: 0.2334	Loss_G: 3.1728	D(x): 0.8654	D(G(z)): 0.0729 / 0.0584
  <br />  [8/20][850/1583]	Loss_D: 0.3885	Loss_G: 3.6571	D(x): 0.9005	D(G(z)): 0.2243 / 0.0359
  <br />  [8/20][900/1583]	Loss_D: 0.4124	Loss_G: 3.9195	D(x): 0.9031	D(G(z)): 0.2408 / 0.0297
  <br />  [8/20][950/1583]	Loss_D: 0.5008	Loss_G: 2.7880	D(x): 0.8524	D(G(z)): 0.2472 / 0.0879
  <br />  [8/20][1000/1583]	Loss_D: 0.4570	Loss_G: 3.8953	D(x): 0.9260	D(G(z)): 0.2862 / 0.0278
  <br />  [8/20][1050/1583]	Loss_D: 3.3027	Loss_G: 0.0994	D(x): 0.0634	D(G(z)): 0.0111 / 0.9114
  <br />  [8/20][1100/1583]	Loss_D: 0.5205	Loss_G: 1.8326	D(x): 0.7018	D(G(z)): 0.1072 / 0.1995
  <br />  [8/20][1150/1583]	Loss_D: 0.4758	Loss_G: 4.3610	D(x): 0.9459	D(G(z)): 0.2994 / 0.0198
  <br />  [8/20][1200/1583]	Loss_D: 1.1320	Loss_G: 0.4335	D(x): 0.3994	D(G(z)): 0.0359 / 0.6783
  <br />  [8/20][1250/1583]	Loss_D: 0.2684	Loss_G: 3.7708	D(x): 0.9262	D(G(z)): 0.1600 / 0.0331
  <br />  [8/20][1300/1583]	Loss_D: 0.3112	Loss_G: 2.8636	D(x): 0.8433	D(G(z)): 0.1063 / 0.0870
  <br />  [8/20][1350/1583]	Loss_D: 0.3549	Loss_G: 3.3965	D(x): 0.8527	D(G(z)): 0.1540 / 0.0511
  <br />  [8/20][1400/1583]	Loss_D: 0.3796	Loss_G: 1.9934	D(x): 0.7987	D(G(z)): 0.1194 / 0.1735
  <br />  [8/20][1450/1583]	Loss_D: 0.3327	Loss_G: 2.9824	D(x): 0.9003	D(G(z)): 0.1794 / 0.0725
  <br />  [8/20][1500/1583]	Loss_D: 0.8892	Loss_G: 0.9814	D(x): 0.4607	D(G(z)): 0.0146 / 0.4348
  <br />  [8/20][1550/1583]	Loss_D: 0.4178	Loss_G: 2.1137	D(x): 0.7210	D(G(z)): 0.0524 / 0.1642
  <br />  [9/20][0/1583]	Loss_D: 0.3355	Loss_G: 3.4654	D(x): 0.8873	D(G(z)): 0.1723 / 0.0433
  <br />  [9/20][50/1583]	Loss_D: 0.3833	Loss_G: 3.7069	D(x): 0.8648	D(G(z)): 0.1832 / 0.0353
  <br />  [9/20][100/1583]	Loss_D: 0.4653	Loss_G: 3.7419	D(x): 0.9129	D(G(z)): 0.2668 / 0.0393
  <br />  [9/20][150/1583]	Loss_D: 0.3401	Loss_G: 2.4734	D(x): 0.8479	D(G(z)): 0.1401 / 0.1138
  <br />  [9/20][200/1583]	Loss_D: 2.3238	Loss_G: 4.4457	D(x): 0.9894	D(G(z)): 0.8347 / 0.0216
  <br />  [9/20][250/1583]	Loss_D: 0.5899	Loss_G: 1.2450	D(x): 0.6232	D(G(z)): 0.0364 / 0.3604
  <br />  [9/20][300/1583]	Loss_D: 0.6175	Loss_G: 2.7579	D(x): 0.7624	D(G(z)): 0.2380 / 0.0926
  <br />  [9/20][350/1583]	Loss_D: 1.1463	Loss_G: 0.6558	D(x): 0.3963	D(G(z)): 0.0196 / 0.5618
  <br />  [9/20][400/1583]	Loss_D: 0.3826	Loss_G: 2.1367	D(x): 0.7656	D(G(z)): 0.0702 / 0.1567
  <br />  [9/20][450/1583]	Loss_D: 0.8569	Loss_G: 4.9679	D(x): 0.9560	D(G(z)): 0.4928 / 0.0102
  <br />  [9/20][500/1583]	Loss_D: 1.0243	Loss_G: 3.8232	D(x): 0.9137	D(G(z)): 0.5489 / 0.0374
  <br />  [9/20][550/1583]	Loss_D: 0.4581	Loss_G: 3.4023	D(x): 0.8716	D(G(z)): 0.2388 / 0.0464
  <br />  [9/20][600/1583]	Loss_D: 0.2323	Loss_G: 3.0196	D(x): 0.8663	D(G(z)): 0.0747 / 0.0707
  <br />  [9/20][650/1583]	Loss_D: 0.4908	Loss_G: 3.6812	D(x): 0.8512	D(G(z)): 0.2212 / 0.0431
  <br />  [9/20][700/1583]	Loss_D: 0.7221	Loss_G: 5.3467	D(x): 0.9737	D(G(z)): 0.4437 / 0.0077
  <br />  [9/20][750/1583]	Loss_D: 0.3680	Loss_G: 2.6479	D(x): 0.8199	D(G(z)): 0.1334 / 0.0943
  <br />  [9/20][800/1583]	Loss_D: 0.3675	Loss_G: 2.0510	D(x): 0.7923	D(G(z)): 0.1031 / 0.1625
  <br />  [9/20][850/1583]	Loss_D: 0.3144	Loss_G: 4.3019	D(x): 0.9633	D(G(z)): 0.2227 / 0.0192
  <br />  [9/20][900/1583]	Loss_D: 1.6036	Loss_G: 0.8945	D(x): 0.2758	D(G(z)): 0.0344 / 0.4773
  <br />  [9/20][950/1583]	Loss_D: 0.4085	Loss_G: 3.2953	D(x): 0.8953	D(G(z)): 0.2265 / 0.0503
  <br />  [9/20][1000/1583]	Loss_D: 0.3107	Loss_G: 3.3860	D(x): 0.8849	D(G(z)): 0.1489 / 0.0550
  <br />  [9/20][1050/1583]	Loss_D: 0.3532	Loss_G: 2.9715	D(x): 0.8766	D(G(z)): 0.1762 / 0.0710
  <br />  [9/20][1100/1583]	Loss_D: 0.5160	Loss_G: 3.6577	D(x): 0.8796	D(G(z)): 0.2767 / 0.0425
  <br />  [9/20][1150/1583]	Loss_D: 1.3764	Loss_G: 4.0215	D(x): 0.9326	D(G(z)): 0.6596 / 0.0311
  <br />  [9/20][1200/1583]	Loss_D: 0.3632	Loss_G: 3.3313	D(x): 0.8521	D(G(z)): 0.1523 / 0.0535
  <br />  [9/20][1250/1583]	Loss_D: 0.2771	Loss_G: 3.2974	D(x): 0.8931	D(G(z)): 0.1350 / 0.0510
  <br />  [9/20][1300/1583]	Loss_D: 1.0717	Loss_G: 0.5304	D(x): 0.4159	D(G(z)): 0.0438 / 0.6390
  <br />  [9/20][1350/1583]	Loss_D: 0.2916	Loss_G: 2.1300	D(x): 0.8480	D(G(z)): 0.1016 / 0.1599
  <br />  [9/20][1400/1583]	Loss_D: 0.4418	Loss_G: 2.5393	D(x): 0.7389	D(G(z)): 0.0926 / 0.1136
  <br />  [9/20][1450/1583]	Loss_D: 0.4443	Loss_G: 2.2956	D(x): 0.7454	D(G(z)): 0.0954 / 0.1465
  <br />  [9/20][1500/1583]	Loss_D: 0.2820	Loss_G: 3.3872	D(x): 0.8830	D(G(z)): 0.1264 / 0.0512
  <br />  [9/20][1550/1583]	Loss_D: 0.2550	Loss_G: 3.5616	D(x): 0.9287	D(G(z)): 0.1501 / 0.0430
  <br />  [10/20][0/1583]	Loss_D: 0.4452	Loss_G: 5.2619	D(x): 0.9795	D(G(z)): 0.3109 / 0.0078
  <br />  [10/20][50/1583]	Loss_D: 0.3154	Loss_G: 5.1106	D(x): 0.9350	D(G(z)): 0.2016 / 0.0094
  <br />  [10/20][100/1583]	Loss_D: 0.8687	Loss_G: 2.1417	D(x): 0.6353	D(G(z)): 0.2446 / 0.1601
  <br />  [10/20][150/1583]	Loss_D: 0.3782	Loss_G: 2.2331	D(x): 0.7631	D(G(z)): 0.0646 / 0.1492
  <br />  [10/20][200/1583]	Loss_D: 0.3533	Loss_G: 5.1649	D(x): 0.9514	D(G(z)): 0.2351 / 0.0090
  <br />  [10/20][250/1583]	Loss_D: 0.3042	Loss_G: 4.2050	D(x): 0.9426	D(G(z)): 0.1991 / 0.0204
  <br />  [10/20][300/1583]	Loss_D: 0.4181	Loss_G: 3.1368	D(x): 0.9099	D(G(z)): 0.2471 / 0.0611
  <br />  [10/20][350/1583]	Loss_D: 0.3003	Loss_G: 3.2192	D(x): 0.8692	D(G(z)): 0.1221 / 0.0579
  <br />  [10/20][400/1583]	Loss_D: 0.1929	Loss_G: 3.8826	D(x): 0.9587	D(G(z)): 0.1306 / 0.0304
  <br />  [10/20][450/1583]	Loss_D: 0.3430	Loss_G: 3.6889	D(x): 0.9103	D(G(z)): 0.1918 / 0.0369
  <br />  [10/20][500/1583]	Loss_D: 0.1794	Loss_G: 3.4852	D(x): 0.9511	D(G(z)): 0.1145 / 0.0421
  <br />  [10/20][550/1583]	Loss_D: 0.4598	Loss_G: 3.7130	D(x): 0.8519	D(G(z)): 0.2169 / 0.0373
  <br />  [10/20][600/1583]	Loss_D: 0.8685	Loss_G: 2.0182	D(x): 0.5017	D(G(z)): 0.0311 / 0.1981
  <br />  [10/20][650/1583]	Loss_D: 0.3008	Loss_G: 3.7633	D(x): 0.9257	D(G(z)): 0.1768 / 0.0332
  <br />  [10/20][700/1583]	Loss_D: 0.4212	Loss_G: 2.5866	D(x): 0.7214	D(G(z)): 0.0412 / 0.1195
  <br />  [10/20][750/1583]	Loss_D: 0.4810	Loss_G: 2.3168	D(x): 0.6815	D(G(z)): 0.0369 / 0.1446
  <br />  [10/20][800/1583]	Loss_D: 0.3565	Loss_G: 3.1685	D(x): 0.7729	D(G(z)): 0.0687 / 0.0678
  <br />  [10/20][850/1583]	Loss_D: 0.2442	Loss_G: 3.2289	D(x): 0.9054	D(G(z)): 0.1184 / 0.0539
  <br />  [10/20][900/1583]	Loss_D: 0.5866	Loss_G: 4.6595	D(x): 0.9247	D(G(z)): 0.3470 / 0.0133
  <br />  [10/20][950/1583]	Loss_D: 0.2967	Loss_G: 4.1433	D(x): 0.9466	D(G(z)): 0.1993 / 0.0218
  <br />  [10/20][1000/1583]	Loss_D: 0.3342	Loss_G: 2.7950	D(x): 0.8801	D(G(z)): 0.1627 / 0.0827
  <br />  [10/20][1050/1583]	Loss_D: 0.2174	Loss_G: 4.5498	D(x): 0.9690	D(G(z)): 0.1555 / 0.0154
  <br />  [10/20][1100/1583]	Loss_D: 1.0542	Loss_G: 2.9927	D(x): 0.7132	D(G(z)): 0.3997 / 0.0745
  <br />  [10/20][1150/1583]	Loss_D: 1.7374	Loss_G: 4.8391	D(x): 0.9652	D(G(z)): 0.7398 / 0.0180
  <br />  [10/20][1200/1583]	Loss_D: 0.9674	Loss_G: 6.0433	D(x): 0.9941	D(G(z)): 0.5405 / 0.0033
  <br />  [10/20][1250/1583]	Loss_D: 1.1832	Loss_G: 4.7771	D(x): 0.9796	D(G(z)): 0.6156 / 0.0152
  <br />  [10/20][1300/1583]	Loss_D: 0.2012	Loss_G: 4.0343	D(x): 0.9415	D(G(z)): 0.1175 / 0.0273
  <br />  [10/20][1350/1583]	Loss_D: 0.2380	Loss_G: 3.4502	D(x): 0.8696	D(G(z)): 0.0748 / 0.0501
  <br />  [10/20][1400/1583]	Loss_D: 0.7133	Loss_G: 5.9600	D(x): 0.9878	D(G(z)): 0.4375 / 0.0051
  <br />  [10/20][1450/1583]	Loss_D: 0.2545	Loss_G: 3.9354	D(x): 0.9061	D(G(z)): 0.1249 / 0.0282
  <br />  [10/20][1500/1583]	Loss_D: 0.3176	Loss_G: 3.3065	D(x): 0.9375	D(G(z)): 0.2004 / 0.0544
  <br />  [10/20][1550/1583]	Loss_D: 0.2906	Loss_G: 4.1556	D(x): 0.9045	D(G(z)): 0.1512 / 0.0243
  <br />  [11/20][0/1583]	Loss_D: 0.4420	Loss_G: 5.3689	D(x): 0.9729	D(G(z)): 0.2932 / 0.0072
  <br />  [11/20][50/1583]	Loss_D: 0.2171	Loss_G: 3.1759	D(x): 0.8906	D(G(z)): 0.0813 / 0.0646
  <br />  [11/20][100/1583]	Loss_D: 0.4129	Loss_G: 1.5819	D(x): 0.7234	D(G(z)): 0.0346 / 0.2740
  <br />  [11/20][150/1583]	Loss_D: 0.2769	Loss_G: 4.0531	D(x): 0.9648	D(G(z)): 0.1988 / 0.0234
  <br />  [11/20][200/1583]	Loss_D: 0.1899	Loss_G: 4.2448	D(x): 0.9554	D(G(z)): 0.1256 / 0.0199
  <br />  [11/20][250/1583]	Loss_D: 5.5494	Loss_G: 7.5674	D(x): 0.9936	D(G(z)): 0.9849 / 0.0012
  <br />  [11/20][300/1583]	Loss_D: 0.5565	Loss_G: 5.1821	D(x): 0.9543	D(G(z)): 0.3435 / 0.0087
  <br />  [11/20][350/1583]	Loss_D: 0.3600	Loss_G: 3.7507	D(x): 0.9017	D(G(z)): 0.1938 / 0.0358
  <br />  [11/20][400/1583]	Loss_D: 0.2291	Loss_G: 3.8418	D(x): 0.9366	D(G(z)): 0.1390 / 0.0305
  <br />  [11/20][450/1583]	Loss_D: 1.1550	Loss_G: 5.3676	D(x): 0.9219	D(G(z)): 0.5864 / 0.0076
  <br />  [11/20][500/1583]	Loss_D: 0.3543	Loss_G: 3.3118	D(x): 0.8499	D(G(z)): 0.1438 / 0.0554
  <br />  [11/20][550/1583]	Loss_D: 0.1973	Loss_G: 3.6824	D(x): 0.9525	D(G(z)): 0.1272 / 0.0375
  <br />  [11/20][600/1583]	Loss_D: 0.1476	Loss_G: 3.9272	D(x): 0.9360	D(G(z)): 0.0712 / 0.0294
  <br />  [11/20][650/1583]	Loss_D: 0.7400	Loss_G: 1.1222	D(x): 0.5568	D(G(z)): 0.0079 / 0.4088
  <br />  [11/20][700/1583]	Loss_D: 1.0109	Loss_G: 6.3587	D(x): 0.9718	D(G(z)): 0.5298 / 0.0036
  <br />  [11/20][750/1583]	Loss_D: 0.2211	Loss_G: 4.0706	D(x): 0.9805	D(G(z)): 0.1694 / 0.0237
  <br />  [11/20][800/1583]	Loss_D: 0.5770	Loss_G: 3.7492	D(x): 0.8907	D(G(z)): 0.3026 / 0.0366
  <br />  [11/20][850/1583]	Loss_D: 0.5977	Loss_G: 6.1048	D(x): 0.9798	D(G(z)): 0.3782 / 0.0032
  <br />  [11/20][900/1583]	Loss_D: 0.1887	Loss_G: 4.0871	D(x): 0.9325	D(G(z)): 0.1003 / 0.0241
  <br />  [11/20][950/1583]	Loss_D: 0.5064	Loss_G: 3.1961	D(x): 0.8477	D(G(z)): 0.2475 / 0.0605
  <br />  [11/20][1000/1583]	Loss_D: 0.1634	Loss_G: 3.6051	D(x): 0.9100	D(G(z)): 0.0578 / 0.0415
  <br />  [11/20][1050/1583]	Loss_D: 0.3455	Loss_G: 4.2352	D(x): 0.9146	D(G(z)): 0.1848 / 0.0219
  <br />  [11/20][1100/1583]	Loss_D: 0.2474	Loss_G: 2.9496	D(x): 0.8627	D(G(z)): 0.0809 / 0.0845
  <br />  [11/20][1150/1583]	Loss_D: 0.2832	Loss_G: 3.0775	D(x): 0.8603	D(G(z)): 0.0916 / 0.0668
  <br />  [11/20][1200/1583]	Loss_D: 0.2517	Loss_G: 5.3835	D(x): 0.9611	D(G(z)): 0.1707 / 0.0072
  <br />  [11/20][1250/1583]	Loss_D: 0.3220	Loss_G: 2.7440	D(x): 0.7723	D(G(z)): 0.0204 / 0.0998
  <br />  [11/20][1300/1583]	Loss_D: 0.1644	Loss_G: 4.6127	D(x): 0.9614	D(G(z)): 0.1071 / 0.0140
  <br />  [11/20][1350/1583]	Loss_D: 0.8478	Loss_G: 1.5851	D(x): 0.5446	D(G(z)): 0.1176 / 0.2687
  <br />  [11/20][1400/1583]	Loss_D: 1.1538	Loss_G: 4.2830	D(x): 0.9453	D(G(z)): 0.5946 / 0.0220
  <br />  [11/20][1450/1583]	Loss_D: 2.3230	Loss_G: 0.3363	D(x): 0.1518	D(G(z)): 0.0075 / 0.7654
  <br />  [11/20][1500/1583]	Loss_D: 0.2925	Loss_G: 5.1475	D(x): 0.9659	D(G(z)): 0.2077 / 0.0081
  <br />  [11/20][1550/1583]	Loss_D: 0.9419	Loss_G: 3.6552	D(x): 0.9646	D(G(z)): 0.5320 / 0.0453
  <br />  [12/20][0/1583]	Loss_D: 0.2329	Loss_G: 3.5086	D(x): 0.9025	D(G(z)): 0.1089 / 0.0459
  <br />  [12/20][50/1583]	Loss_D: 0.2044	Loss_G: 2.7007	D(x): 0.8895	D(G(z)): 0.0736 / 0.0898
  <br />  [12/20][100/1583]	Loss_D: 0.1413	Loss_G: 3.5854	D(x): 0.9585	D(G(z)): 0.0867 / 0.0397
  <br />  [12/20][150/1583]	Loss_D: 0.4105	Loss_G: 2.9610	D(x): 0.9645	D(G(z)): 0.2726 / 0.0753
  <br />  [12/20][200/1583]	Loss_D: 0.1394	Loss_G: 3.6151	D(x): 0.9555	D(G(z)): 0.0807 / 0.0463
  <br />  [12/20][250/1583]	Loss_D: 0.3895	Loss_G: 4.1200	D(x): 0.9206	D(G(z)): 0.2326 / 0.0242
  <br />  [12/20][300/1583]	Loss_D: 0.1476	Loss_G: 3.8114	D(x): 0.9295	D(G(z)): 0.0670 / 0.0332
  <br />  [12/20][350/1583]	Loss_D: 0.2177	Loss_G: 3.9375	D(x): 0.9628	D(G(z)): 0.1479 / 0.0330
  <br />  [12/20][400/1583]	Loss_D: 1.7412	Loss_G: 0.6864	D(x): 0.3020	D(G(z)): 0.1364 / 0.5558
  <br />  [12/20][450/1583]	Loss_D: 0.5029	Loss_G: 5.4238	D(x): 0.9648	D(G(z)): 0.3219 / 0.0072
  <br />  [12/20][500/1583]	Loss_D: 0.1640	Loss_G: 3.6570	D(x): 0.9278	D(G(z)): 0.0772 / 0.0417
  <br />  [12/20][550/1583]	Loss_D: 0.1880	Loss_G: 3.7419	D(x): 0.9164	D(G(z)): 0.0822 / 0.0383
  <br />  [12/20][600/1583]	Loss_D: 0.5958	Loss_G: 1.7376	D(x): 0.7388	D(G(z)): 0.2005 / 0.2267
  <br />  [12/20][650/1583]	Loss_D: 0.1479	Loss_G: 3.5536	D(x): 0.9287	D(G(z)): 0.0621 / 0.0427
  <br />  [12/20][700/1583]	Loss_D: 0.4919	Loss_G: 5.9880	D(x): 0.9669	D(G(z)): 0.3191 / 0.0043
  <br />  [12/20][750/1583]	Loss_D: 0.1262	Loss_G: 3.7576	D(x): 0.9288	D(G(z)): 0.0464 / 0.0364
  <br />  [12/20][800/1583]	Loss_D: 0.1352	Loss_G: 4.5882	D(x): 0.9788	D(G(z)): 0.1020 / 0.0152
  <br />  [12/20][850/1583]	Loss_D: 1.1377	Loss_G: 4.6945	D(x): 0.9619	D(G(z)): 0.5560 / 0.0168
  <br />  [12/20][900/1583]	Loss_D: 0.3021	Loss_G: 2.8691	D(x): 0.8326	D(G(z)): 0.0899 / 0.0947
  <br />  [12/20][950/1583]	Loss_D: 3.6154	Loss_G: 0.1904	D(x): 0.0592	D(G(z)): 0.0439 / 0.8445
  <br />  [12/20][1000/1583]	Loss_D: 0.9654	Loss_G: 1.2462	D(x): 0.5309	D(G(z)): 0.1794 / 0.3414
  <br />  [12/20][1050/1583]	Loss_D: 0.5466	Loss_G: 3.0663	D(x): 0.8267	D(G(z)): 0.2374 / 0.0787
  <br />  [12/20][1100/1583]	Loss_D: 0.7550	Loss_G: 5.3458	D(x): 0.9694	D(G(z)): 0.4454 / 0.0077
  <br />  [12/20][1150/1583]	Loss_D: 0.7596	Loss_G: 5.1425	D(x): 0.9850	D(G(z)): 0.4588 / 0.0102
  <br />  [12/20][1200/1583]	Loss_D: 1.8692	Loss_G: 0.9769	D(x): 0.2173	D(G(z)): 0.0100 / 0.4478
  <br />  [12/20][1250/1583]	Loss_D: 0.6342	Loss_G: 2.4452	D(x): 0.6621	D(G(z)): 0.1301 / 0.1391
  <br />  [12/20][1300/1583]	Loss_D: 0.1406	Loss_G: 4.2106	D(x): 0.9654	D(G(z)): 0.0952 / 0.0235
  <br />  [12/20][1350/1583]	Loss_D: 0.2189	Loss_G: 4.2622	D(x): 0.9401	D(G(z)): 0.1320 / 0.0236
  <br />  [12/20][1400/1583]	Loss_D: 0.1422	Loss_G: 3.1437	D(x): 0.9681	D(G(z)): 0.0973 / 0.0636
  <br />  [12/20][1450/1583]	Loss_D: 0.2394	Loss_G: 3.6276	D(x): 0.8526	D(G(z)): 0.0582 / 0.0529
  <br />  [12/20][1500/1583]	Loss_D: 0.3643	Loss_G: 3.0499	D(x): 0.9099	D(G(z)): 0.2069 / 0.0641
  <br />  [12/20][1550/1583]	Loss_D: 0.1814	Loss_G: 3.2492	D(x): 0.9000	D(G(z)): 0.0637 / 0.0601
  <br />  [13/20][0/1583]	Loss_D: 0.1948	Loss_G: 3.0924	D(x): 0.9062	D(G(z)): 0.0800 / 0.0737
  <br />  [13/20][50/1583]	Loss_D: 0.2844	Loss_G: 4.5420	D(x): 0.9692	D(G(z)): 0.1984 / 0.0173
  <br />  [13/20][100/1583]	Loss_D: 0.2022	Loss_G: 4.4813	D(x): 0.9712	D(G(z)): 0.1458 / 0.0169
  <br />  [13/20][150/1583]	Loss_D: 0.2347	Loss_G: 4.6477	D(x): 0.9405	D(G(z)): 0.1446 / 0.0159
  <br />  [13/20][200/1583]	Loss_D: 0.5541	Loss_G: 3.5491	D(x): 0.9130	D(G(z)): 0.3298 / 0.0446
  <br />  [13/20][250/1583]	Loss_D: 0.1553	Loss_G: 3.9486	D(x): 0.9367	D(G(z)): 0.0775 / 0.0319
  <br />  [13/20][300/1583]	Loss_D: 0.2056	Loss_G: 4.4578	D(x): 0.9795	D(G(z)): 0.1509 / 0.0194
  <br />  [13/20][350/1583]	Loss_D: 0.1145	Loss_G: 3.9158	D(x): 0.9419	D(G(z)): 0.0489 / 0.0358
  <br />  [13/20][400/1583]	Loss_D: 0.1810	Loss_G: 3.9198	D(x): 0.9068	D(G(z)): 0.0628 / 0.0342
  <br />  [13/20][450/1583]	Loss_D: 0.1737	Loss_G: 4.3717	D(x): 0.9566	D(G(z)): 0.1110 / 0.0204
  <br />  [13/20][500/1583]	Loss_D: 0.1324	Loss_G: 3.6611	D(x): 0.9361	D(G(z)): 0.0589 / 0.0383
  <br />  [13/20][550/1583]	Loss_D: 0.1221	Loss_G: 4.1953	D(x): 0.9447	D(G(z)): 0.0588 / 0.0233
  <br />  [13/20][600/1583]	Loss_D: 0.4661	Loss_G: 3.5945	D(x): 0.9576	D(G(z)): 0.3072 / 0.0404
  <br />  [13/20][650/1583]	Loss_D: 0.3837	Loss_G: 6.8633	D(x): 0.9804	D(G(z)): 0.2731 / 0.0017
  <br />  [13/20][700/1583]	Loss_D: 0.7979	Loss_G: 3.4169	D(x): 0.8841	D(G(z)): 0.4090 / 0.0527
  <br />  [13/20][750/1583]	Loss_D: 1.2752	Loss_G: 1.2170	D(x): 0.5623	D(G(z)): 0.3042 / 0.3905
  <br />  [13/20][800/1583]	Loss_D: 0.8152	Loss_G: 6.7914	D(x): 0.9930	D(G(z)): 0.4845 / 0.0017
  <br />  [13/20][850/1583]	Loss_D: 0.2118	Loss_G: 4.2853	D(x): 0.9474	D(G(z)): 0.1337 / 0.0215
  <br />  [13/20][900/1583]	Loss_D: 0.2051	Loss_G: 3.4792	D(x): 0.9264	D(G(z)): 0.1058 / 0.0455
  <br />  [13/20][950/1583]	Loss_D: 0.2248	Loss_G: 2.8438	D(x): 0.8539	D(G(z)): 0.0430 / 0.0884
  <br />  [13/20][1000/1583]	Loss_D: 0.2403	Loss_G: 2.5586	D(x): 0.8343	D(G(z)): 0.0397 / 0.1121
  <br />  [13/20][1050/1583]	Loss_D: 0.1286	Loss_G: 3.0089	D(x): 0.9641	D(G(z)): 0.0807 / 0.0763
  <br />  [13/20][1100/1583]	Loss_D: 0.5571	Loss_G: 3.4412	D(x): 0.9439	D(G(z)): 0.3437 / 0.0475
  <br />  [13/20][1150/1583]	Loss_D: 0.2479	Loss_G: 4.0702	D(x): 0.9362	D(G(z)): 0.1472 / 0.0280
  <br />  [13/20][1200/1583]	Loss_D: 0.4653	Loss_G: 2.1317	D(x): 0.7067	D(G(z)): 0.0458 / 0.1941
  <br />  [13/20][1250/1583]	Loss_D: 0.1503	Loss_G: 4.3328	D(x): 0.9729	D(G(z)): 0.1086 / 0.0189
  <br />  [13/20][1300/1583]	Loss_D: 0.1317	Loss_G: 3.9708	D(x): 0.9537	D(G(z)): 0.0671 / 0.0300
  <br />  [13/20][1350/1583]	Loss_D: 0.1118	Loss_G: 3.9194	D(x): 0.9500	D(G(z)): 0.0534 / 0.0319
  <br />  [13/20][1400/1583]	Loss_D: 0.1906	Loss_G: 3.5532	D(x): 0.8952	D(G(z)): 0.0578 / 0.0519
  <br />  [13/20][1450/1583]	Loss_D: 0.9284	Loss_G: 1.2762	D(x): 0.5485	D(G(z)): 0.1624 / 0.3366
  <br />  [13/20][1500/1583]	Loss_D: 0.9241	Loss_G: 1.8346	D(x): 0.5324	D(G(z)): 0.1070 / 0.2368
  <br />  [13/20][1550/1583]	Loss_D: 0.2680	Loss_G: 3.7866	D(x): 0.8293	D(G(z)): 0.0453 / 0.0422
  <br />  [14/20][0/1583]	Loss_D: 0.1989	Loss_G: 4.7444	D(x): 0.9594	D(G(z)): 0.1353 / 0.0127
  <br />  [14/20][50/1583]	Loss_D: 0.1971	Loss_G: 2.5916	D(x): 0.8815	D(G(z)): 0.0576 / 0.1124
  <br />  [14/20][100/1583]	Loss_D: 0.1191	Loss_G: 3.8794	D(x): 0.9612	D(G(z)): 0.0724 / 0.0299
  <br />  [14/20][150/1583]	Loss_D: 0.1835	Loss_G: 5.2542	D(x): 0.9788	D(G(z)): 0.1383 / 0.0083
  <br />  [14/20][200/1583]	Loss_D: 0.3187	Loss_G: 3.5453	D(x): 0.8965	D(G(z)): 0.1660 / 0.0436
  <br />  [14/20][250/1583]	Loss_D: 0.1011	Loss_G: 4.4302	D(x): 0.9554	D(G(z)): 0.0486 / 0.0182
  <br />  [14/20][300/1583]	Loss_D: 0.1254	Loss_G: 3.6950	D(x): 0.9370	D(G(z)): 0.0537 / 0.0402
  <br />  [14/20][350/1583]	Loss_D: 0.3753	Loss_G: 2.9771	D(x): 0.7759	D(G(z)): 0.0636 / 0.0855
  <br />  [14/20][400/1583]	Loss_D: 1.8167	Loss_G: 1.0710	D(x): 0.2214	D(G(z)): 0.0314 / 0.4136
  <br />  [14/20][450/1583]	Loss_D: 0.5008	Loss_G: 2.8074	D(x): 0.7281	D(G(z)): 0.1087 / 0.0906
  <br />  [14/20][500/1583]	Loss_D: 0.2605	Loss_G: 2.9950	D(x): 0.8243	D(G(z)): 0.0443 / 0.0843
  <br />  [14/20][550/1583]	Loss_D: 0.1909	Loss_G: 4.0212	D(x): 0.9111	D(G(z)): 0.0764 / 0.0298
  <br />  [14/20][600/1583]	Loss_D: 0.2020	Loss_G: 2.3619	D(x): 0.8611	D(G(z)): 0.0350 / 0.1540
  <br />  [14/20][650/1583]	Loss_D: 0.2275	Loss_G: 2.9458	D(x): 0.8885	D(G(z)): 0.0928 / 0.0741
  <br />  [14/20][700/1583]	Loss_D: 0.0784	Loss_G: 4.4522	D(x): 0.9726	D(G(z)): 0.0462 / 0.0207
  <br />  [14/20][750/1583]	Loss_D: 0.0868	Loss_G: 3.5306	D(x): 0.9504	D(G(z)): 0.0325 / 0.0493
  <br />  [14/20][800/1583]	Loss_D: 0.1172	Loss_G: 3.4457	D(x): 0.9258	D(G(z)): 0.0311 / 0.0497
  <br />  [14/20][850/1583]	Loss_D: 0.1150	Loss_G: 4.7514	D(x): 0.9659	D(G(z)): 0.0681 / 0.0138
  <br />  [14/20][900/1583]	Loss_D: 0.6961	Loss_G: 1.0161	D(x): 0.5965	D(G(z)): 0.0307 / 0.4432
  <br />  [14/20][950/1583]	Loss_D: 0.5427	Loss_G: 2.5343	D(x): 0.6771	D(G(z)): 0.0792 / 0.1282
  <br />  [14/20][1000/1583]	Loss_D: 0.8422	Loss_G: 4.1118	D(x): 0.9256	D(G(z)): 0.4544 / 0.0273
  <br />  [14/20][1050/1583]	Loss_D: 1.2674	Loss_G: 5.3719	D(x): 0.9754	D(G(z)): 0.6236 / 0.0107
  <br />  [14/20][1100/1583]	Loss_D: 0.1675	Loss_G: 3.5696	D(x): 0.8867	D(G(z)): 0.0351 / 0.0474
  <br />  [14/20][1150/1583]	Loss_D: 0.1116	Loss_G: 3.9747	D(x): 0.9553	D(G(z)): 0.0592 / 0.0319
  <br />  [14/20][1200/1583]	Loss_D: 0.6670	Loss_G: 2.2726	D(x): 0.6947	D(G(z)): 0.1970 / 0.1443
  <br />  [14/20][1250/1583]	Loss_D: 0.1379	Loss_G: 4.7130	D(x): 0.9647	D(G(z)): 0.0851 / 0.0168
  <br />  [14/20][1300/1583]	Loss_D: 0.1290	Loss_G: 3.9698	D(x): 0.9321	D(G(z)): 0.0501 / 0.0351
  <br />  [14/20][1350/1583]	Loss_D: 0.1147	Loss_G: 3.5459	D(x): 0.9406	D(G(z)): 0.0484 / 0.0436
  <br />  [14/20][1400/1583]	Loss_D: 1.3923	Loss_G: 1.5737	D(x): 0.6575	D(G(z)): 0.4338 / 0.3123
  <br />  [14/20][1450/1583]	Loss_D: 0.5742	Loss_G: 2.1677	D(x): 0.6605	D(G(z)): 0.0646 / 0.1866
  <br />  [14/20][1500/1583]	Loss_D: 0.4496	Loss_G: 3.5803	D(x): 0.8910	D(G(z)): 0.2410 / 0.0469
  <br />  [14/20][1550/1583]	Loss_D: 0.2170	Loss_G: 3.1385	D(x): 0.8570	D(G(z)): 0.0334 / 0.0682
  <br />  [15/20][0/1583]	Loss_D: 1.2796	Loss_G: 3.3778	D(x): 0.8812	D(G(z)): 0.5999 / 0.0576
  <br />  [15/20][50/1583]	Loss_D: 0.1698	Loss_G: 3.5684	D(x): 0.9126	D(G(z)): 0.0629 / 0.0483
  <br />  [15/20][100/1583]	Loss_D: 0.2076	Loss_G: 3.1623	D(x): 0.8790	D(G(z)): 0.0629 / 0.0730
  <br />  [15/20][150/1583]	Loss_D: 0.1487	Loss_G: 4.5114	D(x): 0.8858	D(G(z)): 0.0096 / 0.0205
  <br />  [15/20][200/1583]	Loss_D: 0.1168	Loss_G: 3.9740	D(x): 0.9295	D(G(z)): 0.0341 / 0.0319
  <br />  [15/20][250/1583]	Loss_D: 0.1519	Loss_G: 4.3522	D(x): 0.9498	D(G(z)): 0.0826 / 0.0225
  <br />  [15/20][300/1583]	Loss_D: 0.1599	Loss_G: 3.2387	D(x): 0.8821	D(G(z)): 0.0215 / 0.0672
  <br />  [15/20][350/1583]	Loss_D: 0.1400	Loss_G: 5.3293	D(x): 0.9823	D(G(z)): 0.1030 / 0.0078
  <br />  [15/20][400/1583]	Loss_D: 0.1654	Loss_G: 4.2977	D(x): 0.9338	D(G(z)): 0.0799 / 0.0245
  <br />  [15/20][450/1583]	Loss_D: 0.1649	Loss_G: 4.2507	D(x): 0.9492	D(G(z)): 0.0932 / 0.0248
  <br />  [15/20][500/1583]	Loss_D: 0.3518	Loss_G: 5.5630	D(x): 0.9538	D(G(z)): 0.2218 / 0.0071
  <br />  [15/20][550/1583]	Loss_D: 0.2416	Loss_G: 3.6121	D(x): 0.8917	D(G(z)): 0.0970 / 0.0500
  <br />  [15/20][600/1583]	Loss_D: 0.9827	Loss_G: 3.2378	D(x): 0.8402	D(G(z)): 0.4914 / 0.0543
  <br />  [15/20][650/1583]	Loss_D: 0.7927	Loss_G: 1.4657	D(x): 0.6582	D(G(z)): 0.2021 / 0.2974
  <br />  [15/20][700/1583]	Loss_D: 0.2597	Loss_G: 6.1874	D(x): 0.9747	D(G(z)): 0.1770 / 0.0041
  <br />  [15/20][750/1583]	Loss_D: 0.6341	Loss_G: 4.5647	D(x): 0.9886	D(G(z)): 0.3928 / 0.0161
  <br />  [15/20][800/1583]	Loss_D: 0.1049	Loss_G: 4.0093	D(x): 0.9326	D(G(z)): 0.0309 / 0.0294
  <br />  [15/20][850/1583]	Loss_D: 0.2053	Loss_G: 4.5544	D(x): 0.9180	D(G(z)): 0.1014 / 0.0171
  <br />  [15/20][900/1583]	Loss_D: 0.3169	Loss_G: 4.2614	D(x): 0.9107	D(G(z)): 0.1746 / 0.0224
  <br />  [15/20][950/1583]	Loss_D: 0.1002	Loss_G: 4.4715	D(x): 0.9744	D(G(z)): 0.0682 / 0.0183
  <br />  [15/20][1000/1583]	Loss_D: 0.1638	Loss_G: 4.4362	D(x): 0.9780	D(G(z)): 0.1214 / 0.0191
  <br />  [15/20][1050/1583]	Loss_D: 0.1286	Loss_G: 4.5392	D(x): 0.9126	D(G(z)): 0.0289 / 0.0222
  <br />  [15/20][1100/1583]	Loss_D: 0.0945	Loss_G: 4.6884	D(x): 0.9696	D(G(z)): 0.0571 / 0.0160
  <br />  [15/20][1150/1583]	Loss_D: 0.1025	Loss_G: 3.6900	D(x): 0.9474	D(G(z)): 0.0434 / 0.0403
  <br />  [15/20][1200/1583]	Loss_D: 2.4789	Loss_G: 0.4233	D(x): 0.1553	D(G(z)): 0.0047 / 0.6973
  <br />  [15/20][1250/1583]	Loss_D: 0.1330	Loss_G: 4.3373	D(x): 0.9340	D(G(z)): 0.0564 / 0.0228
  <br />  [15/20][1300/1583]	Loss_D: 0.0977	Loss_G: 4.1846	D(x): 0.9572	D(G(z)): 0.0491 / 0.0238
  <br />  [15/20][1350/1583]	Loss_D: 0.4272	Loss_G: 2.1301	D(x): 0.7160	D(G(z)): 0.0252 / 0.2052
  <br />  [15/20][1400/1583]	Loss_D: 3.8339	Loss_G: 0.6287	D(x): 0.0447	D(G(z)): 0.0005 / 0.5831
  <br />  [15/20][1450/1583]	Loss_D: 0.9413	Loss_G: 3.1921	D(x): 0.6937	D(G(z)): 0.3047 / 0.1274
  <br />  [15/20][1500/1583]	Loss_D: 0.2445	Loss_G: 4.2717	D(x): 0.9180	D(G(z)): 0.1208 / 0.0259
  <br />  [15/20][1550/1583]	Loss_D: 0.1066	Loss_G: 4.0752	D(x): 0.9529	D(G(z)): 0.0521 / 0.0286
  <br />  [16/20][0/1583]	Loss_D: 0.1498	Loss_G: 4.7893	D(x): 0.9734	D(G(z)): 0.1037 / 0.0132
  <br />  [16/20][50/1583]	Loss_D: 0.1188	Loss_G: 4.0038	D(x): 0.9538	D(G(z)): 0.0629 / 0.0295
  <br />  [16/20][100/1583]	Loss_D: 0.1006	Loss_G: 4.1583	D(x): 0.9356	D(G(z)): 0.0263 / 0.0293
  <br />  [16/20][150/1583]	Loss_D: 0.7203	Loss_G: 6.8572	D(x): 0.9780	D(G(z)): 0.4173 / 0.0021
  <br />  [16/20][200/1583]	Loss_D: 0.0920	Loss_G: 3.9698	D(x): 0.9419	D(G(z)): 0.0282 / 0.0378
  <br />  [16/20][250/1583]	Loss_D: 0.6379	Loss_G: 8.9561	D(x): 0.9979	D(G(z)): 0.4009 / 0.0003
  <br />  [16/20][300/1583]	Loss_D: 0.6091	Loss_G: 3.3029	D(x): 0.8244	D(G(z)): 0.2925 / 0.0539
  <br />  [16/20][350/1583]	Loss_D: 4.3029	Loss_G: 7.3242	D(x): 0.9886	D(G(z)): 0.9378 / 0.0029
  <br />  [16/20][400/1583]	Loss_D: 2.0085	Loss_G: 0.9957	D(x): 0.2677	D(G(z)): 0.1036 / 0.4792
  <br />  [16/20][450/1583]	Loss_D: 0.6552	Loss_G: 2.6439	D(x): 0.6444	D(G(z)): 0.0637 / 0.1108
  <br />  [16/20][500/1583]	Loss_D: 0.3052	Loss_G: 2.6446	D(x): 0.8240	D(G(z)): 0.0669 / 0.1124
  <br />  [16/20][550/1583]	Loss_D: 0.1454	Loss_G: 5.1004	D(x): 0.9791	D(G(z)): 0.1087 / 0.0089
  <br />  [16/20][600/1583]	Loss_D: 0.2425	Loss_G: 4.7075	D(x): 0.9728	D(G(z)): 0.1730 / 0.0139
  <br />  [16/20][650/1583]	Loss_D: 0.1498	Loss_G: 5.8204	D(x): 0.9928	D(G(z)): 0.1251 / 0.0047
  <br />  [16/20][700/1583]	Loss_D: 0.1259	Loss_G: 5.5439	D(x): 0.9864	D(G(z)): 0.0981 / 0.0062
  <br />  [16/20][750/1583]	Loss_D: 0.0915	Loss_G: 3.7784	D(x): 0.9566	D(G(z)): 0.0431 / 0.0354
  <br />  [16/20][800/1583]	Loss_D: 0.2413	Loss_G: 5.6368	D(x): 0.9798	D(G(z)): 0.1697 / 0.0059
  <br />  [16/20][850/1583]	Loss_D: 0.2382	Loss_G: 7.1151	D(x): 0.9910	D(G(z)): 0.1791 / 0.0014
  <br />  [16/20][900/1583]	Loss_D: 0.6640	Loss_G: 7.8861	D(x): 0.9967	D(G(z)): 0.4225 / 0.0007
  <br />  [16/20][950/1583]	Loss_D: 1.8672	Loss_G: 1.0168	D(x): 0.2395	D(G(z)): 0.0556 / 0.4409
  <br />  [16/20][1000/1583]	Loss_D: 0.2520	Loss_G: 4.3822	D(x): 0.9286	D(G(z)): 0.1402 / 0.0238
  <br />  [16/20][1050/1583]	Loss_D: 0.0967	Loss_G: 4.2129	D(x): 0.9404	D(G(z)): 0.0291 / 0.0291
  <br />  [16/20][1100/1583]	Loss_D: 0.1344	Loss_G: 3.7313	D(x): 0.8952	D(G(z)): 0.0109 / 0.0415
  <br />  [16/20][1150/1583]	Loss_D: 0.1069	Loss_G: 3.4631	D(x): 0.9329	D(G(z)): 0.0310 / 0.0542
  <br />  [16/20][1200/1583]	Loss_D: 0.1169	Loss_G: 4.4238	D(x): 0.9630	D(G(z)): 0.0691 / 0.0211
  <br />  [16/20][1250/1583]	Loss_D: 0.1653	Loss_G: 3.5812	D(x): 0.9259	D(G(z)): 0.0671 / 0.0469
  <br />  [16/20][1300/1583]	Loss_D: 0.8881	Loss_G: 4.5451	D(x): 0.9112	D(G(z)): 0.4803 / 0.0160
  <br />  [16/20][1350/1583]	Loss_D: 0.1268	Loss_G: 4.0125	D(x): 0.9098	D(G(z)): 0.0249 / 0.0344
  <br />  [16/20][1400/1583]	Loss_D: 0.1863	Loss_G: 3.1499	D(x): 0.8899	D(G(z)): 0.0525 / 0.0709
  <br />  [16/20][1450/1583]	Loss_D: 2.3809	Loss_G: 3.3642	D(x): 0.9494	D(G(z)): 0.8505 / 0.0663
  <br />  [16/20][1500/1583]	Loss_D: 0.3161	Loss_G: 2.8555	D(x): 0.8365	D(G(z)): 0.0762 / 0.1079
  <br />  [16/20][1550/1583]	Loss_D: 0.4346	Loss_G: 3.4285	D(x): 0.8773	D(G(z)): 0.2172 / 0.0524
  <br />  [17/20][0/1583]	Loss_D: 0.2392	Loss_G: 4.2136	D(x): 0.9778	D(G(z)): 0.1636 / 0.0273
  <br />  [17/20][50/1583]	Loss_D: 0.1600	Loss_G: 2.9440	D(x): 0.8882	D(G(z)): 0.0296 / 0.0903
  <br />  [17/20][100/1583]	Loss_D: 0.1037	Loss_G: 4.1535	D(x): 0.9437	D(G(z)): 0.0375 / 0.0280
  <br />  [17/20][150/1583]	Loss_D: 0.0924	Loss_G: 4.2635	D(x): 0.9463	D(G(z)): 0.0328 / 0.0255
  <br />  [17/20][200/1583]	Loss_D: 0.0853	Loss_G: 4.4486	D(x): 0.9834	D(G(z)): 0.0580 / 0.0181
  <br />  [17/20][250/1583]	Loss_D: 0.1854	Loss_G: 4.5378	D(x): 0.9576	D(G(z)): 0.1196 / 0.0163
  <br />  [17/20][300/1583]	Loss_D: 0.0713	Loss_G: 4.6036	D(x): 0.9904	D(G(z)): 0.0570 / 0.0166
  <br />  [17/20][350/1583]	Loss_D: 0.0815	Loss_G: 4.6050	D(x): 0.9753	D(G(z)): 0.0480 / 0.0177
  <br />  [17/20][400/1583]	Loss_D: 0.0774	Loss_G: 4.6454	D(x): 0.9582	D(G(z)): 0.0324 / 0.0174
  <br />  [17/20][450/1583]	Loss_D: 0.0776	Loss_G: 4.7100	D(x): 0.9765	D(G(z)): 0.0498 / 0.0158
  <br />  [17/20][500/1583]	Loss_D: 0.0655	Loss_G: 4.3478	D(x): 0.9525	D(G(z)): 0.0143 / 0.0207
  <br />  [17/20][550/1583]	Loss_D: 0.7846	Loss_G: 5.5213	D(x): 0.9698	D(G(z)): 0.4275 / 0.0076
  <br />  [17/20][600/1583]	Loss_D: 0.3724	Loss_G: 4.9101	D(x): 0.9761	D(G(z)): 0.2327 / 0.0129
  <br />  [17/20][650/1583]	Loss_D: 0.1947	Loss_G: 3.8036	D(x): 0.8589	D(G(z)): 0.0167 / 0.0483
  <br />  [17/20][700/1583]	Loss_D: 0.1749	Loss_G: 3.6601	D(x): 0.9395	D(G(z)): 0.0960 / 0.0394
  <br />  [17/20][750/1583]	Loss_D: 0.0840	Loss_G: 4.9581	D(x): 0.9879	D(G(z)): 0.0641 / 0.0127
  <br />  [17/20][800/1583]	Loss_D: 0.0949	Loss_G: 4.9550	D(x): 0.9350	D(G(z)): 0.0222 / 0.0119
  <br />  [17/20][850/1583]	Loss_D: 0.7939	Loss_G: 2.4465	D(x): 0.5609	D(G(z)): 0.0214 / 0.1491
  <br />  [17/20][900/1583]	Loss_D: 0.2218	Loss_G: 2.8206	D(x): 0.8244	D(G(z)): 0.0073 / 0.1055
  <br />  [17/20][950/1583]	Loss_D: 0.6580	Loss_G: 2.1504	D(x): 0.6262	D(G(z)): 0.0473 / 0.1917
  <br />  [17/20][1000/1583]	Loss_D: 0.8152	Loss_G: 7.2704	D(x): 0.9978	D(G(z)): 0.4778 / 0.0014
  <br />  [17/20][1050/1583]	Loss_D: 0.1632	Loss_G: 5.6456	D(x): 0.9865	D(G(z)): 0.1231 / 0.0055
  <br />  [17/20][1100/1583]	Loss_D: 0.1255	Loss_G: 4.7188	D(x): 0.9645	D(G(z)): 0.0771 / 0.0173
  <br />  [17/20][1150/1583]	Loss_D: 0.1923	Loss_G: 3.0381	D(x): 0.8625	D(G(z)): 0.0231 / 0.0849
  <br />  [17/20][1200/1583]	Loss_D: 0.1349	Loss_G: 3.6719	D(x): 0.9055	D(G(z)): 0.0185 / 0.0535
  <br />  [17/20][1250/1583]	Loss_D: 0.0961	Loss_G: 5.2665	D(x): 0.9670	D(G(z)): 0.0554 / 0.0111
  <br />  [17/20][1300/1583]	Loss_D: 0.0753	Loss_G: 4.6870	D(x): 0.9463	D(G(z)): 0.0164 / 0.0189
  <br />  [17/20][1350/1583]	Loss_D: 0.1761	Loss_G: 5.7378	D(x): 0.9731	D(G(z)): 0.1212 / 0.0054
  <br />  [17/20][1400/1583]	Loss_D: 0.4319	Loss_G: 4.6168	D(x): 0.9011	D(G(z)): 0.2229 / 0.0188
  <br />  [17/20][1450/1583]	Loss_D: 1.8176	Loss_G: 1.7620	D(x): 0.2571	D(G(z)): 0.0079 / 0.2608
  <br />  [17/20][1500/1583]	Loss_D: 0.7519	Loss_G: 2.5534	D(x): 0.6595	D(G(z)): 0.1920 / 0.1095
  <br />  [17/20][1550/1583]	Loss_D: 0.3474	Loss_G: 3.1570	D(x): 0.7708	D(G(z)): 0.0265 / 0.0666
  <br />  [18/20][0/1583]	Loss_D: 0.4150	Loss_G: 6.7068	D(x): 0.9909	D(G(z)): 0.2803 / 0.0026
  <br />  [18/20][50/1583]	Loss_D: 0.0937	Loss_G: 4.4016	D(x): 0.9593	D(G(z)): 0.0464 / 0.0216
  <br />  [18/20][100/1583]	Loss_D: 0.1152	Loss_G: 3.8754	D(x): 0.9251	D(G(z)): 0.0278 / 0.0374
  <br />  [18/20][150/1583]	Loss_D: 0.0969	Loss_G: 3.9085	D(x): 0.9341	D(G(z)): 0.0251 / 0.0368
  <br />  [18/20][200/1583]	Loss_D: 0.0734	Loss_G: 4.3257	D(x): 0.9632	D(G(z)): 0.0328 / 0.0215
  <br />  [18/20][250/1583]	Loss_D: 0.0503	Loss_G: 4.2014	D(x): 0.9792	D(G(z)): 0.0278 / 0.0248
  <br />  [18/20][300/1583]	Loss_D: 0.0966	Loss_G: 5.3913	D(x): 0.9919	D(G(z)): 0.0794 / 0.0074
  <br />  [18/20][350/1583]	Loss_D: 0.0304	Loss_G: 5.3741	D(x): 0.9839	D(G(z)): 0.0138 / 0.0083
  <br />  [18/20][400/1583]	Loss_D: 0.0399	Loss_G: 5.1732	D(x): 0.9874	D(G(z)): 0.0258 / 0.0126
  <br />  [18/20][450/1583]	Loss_D: 0.2534	Loss_G: 3.4210	D(x): 0.8515	D(G(z)): 0.0577 / 0.0737
  <br />  [18/20][500/1583]	Loss_D: 0.5689	Loss_G: 2.7608	D(x): 0.8144	D(G(z)): 0.2366 / 0.1026
  <br />  [18/20][550/1583]	Loss_D: 1.6885	Loss_G: 3.8599	D(x): 0.9336	D(G(z)): 0.6982 / 0.0456
  <br />  [18/20][600/1583]	Loss_D: 0.1192	Loss_G: 3.8819	D(x): 0.9474	D(G(z)): 0.0560 / 0.0391
  <br />  [18/20][650/1583]	Loss_D: 0.1555	Loss_G: 4.7784	D(x): 0.9351	D(G(z)): 0.0686 / 0.0197
  <br />  [18/20][700/1583]	Loss_D: 0.0900	Loss_G: 4.8602	D(x): 0.9492	D(G(z)): 0.0317 / 0.0166
  <br />  [18/20][750/1583]	Loss_D: 0.8866	Loss_G: 5.2242	D(x): 0.9657	D(G(z)): 0.4731 / 0.0104
  <br />  [18/20][800/1583]	Loss_D: 0.1804	Loss_G: 4.2431	D(x): 0.9391	D(G(z)): 0.0942 / 0.0281
  <br />  [18/20][850/1583]	Loss_D: 0.0781	Loss_G: 4.6382	D(x): 0.9677	D(G(z)): 0.0384 / 0.0173
  <br />  [18/20][900/1583]	Loss_D: 0.1476	Loss_G: 6.2146	D(x): 0.9865	D(G(z)): 0.1143 / 0.0034
  <br />  [18/20][950/1583]	Loss_D: 0.0834	Loss_G: 4.4089	D(x): 0.9596	D(G(z)): 0.0362 / 0.0230
  <br />  [18/20][1000/1583]	Loss_D: 2.5929	Loss_G: 0.4819	D(x): 0.1635	D(G(z)): 0.0008 / 0.6707
  <br />  [18/20][1050/1583]	Loss_D: 0.1582	Loss_G: 3.9669	D(x): 0.8825	D(G(z)): 0.0202 / 0.0364
  <br />  [18/20][1100/1583]	Loss_D: 0.0735	Loss_G: 4.3646	D(x): 0.9668	D(G(z)): 0.0366 / 0.0225
  <br />  [18/20][1150/1583]	Loss_D: 1.3856	Loss_G: 2.4325	D(x): 0.6476	D(G(z)): 0.3604 / 0.2226
  <br />  [18/20][1200/1583]	Loss_D: 0.2291	Loss_G: 2.3906	D(x): 0.8539	D(G(z)): 0.0511 / 0.1526
  <br />  [18/20][1250/1583]	Loss_D: 1.0133	Loss_G: 1.5649	D(x): 0.6328	D(G(z)): 0.3058 / 0.2654
  <br />  [18/20][1300/1583]	Loss_D: 0.3146	Loss_G: 3.2187	D(x): 0.8018	D(G(z)): 0.0583 / 0.0770
  <br />  [18/20][1350/1583]	Loss_D: 0.8367	Loss_G: 1.1783	D(x): 0.5381	D(G(z)): 0.0149 / 0.3956
  <br />  [18/20][1400/1583]	Loss_D: 0.0882	Loss_G: 4.5715	D(x): 0.9458	D(G(z)): 0.0284 / 0.0173
  <br />  [18/20][1450/1583]	Loss_D: 0.1374	Loss_G: 4.6492	D(x): 0.9769	D(G(z)): 0.0970 / 0.0168
  <br />  [18/20][1500/1583]	Loss_D: 0.0925	Loss_G: 4.1705	D(x): 0.9333	D(G(z)): 0.0196 / 0.0300
  <br />  [18/20][1550/1583]	Loss_D: 0.1096	Loss_G: 4.9891	D(x): 0.9694	D(G(z)): 0.0579 / 0.0147
  <br />  [19/20][0/1583]	Loss_D: 0.0814	Loss_G: 4.2667	D(x): 0.9630	D(G(z)): 0.0395 / 0.0250
  <br />  [19/20][50/1583]	Loss_D: 0.1355	Loss_G: 5.3479	D(x): 0.9750	D(G(z)): 0.0964 / 0.0080
  <br />  [19/20][100/1583]	Loss_D: 0.0680	Loss_G: 4.8436	D(x): 0.9611	D(G(z)): 0.0249 / 0.0152
  <br />  [19/20][150/1583]	Loss_D: 0.1064	Loss_G: 3.1641	D(x): 0.9339	D(G(z)): 0.0288 / 0.0780
  <br />  [19/20][200/1583]	Loss_D: 0.0728	Loss_G: 4.7479	D(x): 0.9545	D(G(z)): 0.0235 / 0.0180
  <br />  [19/20][250/1583]	Loss_D: 0.5200	Loss_G: 2.5695	D(x): 0.8110	D(G(z)): 0.2162 / 0.1155
  <br />  [19/20][300/1583]	Loss_D: 0.1154	Loss_G: 4.3183	D(x): 0.9633	D(G(z)): 0.0698 / 0.0260
  <br />  [19/20][350/1583]	Loss_D: 1.1636	Loss_G: 1.8492	D(x): 0.6942	D(G(z)): 0.4510 / 0.2161
  <br />  [19/20][400/1583]	Loss_D: 0.5549	Loss_G: 3.0729	D(x): 0.7290	D(G(z)): 0.1001 / 0.0819
  <br />  [19/20][450/1583]	Loss_D: 0.2882	Loss_G: 2.2298	D(x): 0.7973	D(G(z)): 0.0201 / 0.1892
  <br />  [19/20][500/1583]	Loss_D: 0.1520	Loss_G: 4.7678	D(x): 0.9477	D(G(z)): 0.0812 / 0.0186
  <br />  [19/20][550/1583]	Loss_D: 0.0880	Loss_G: 3.8464	D(x): 0.9473	D(G(z)): 0.0289 / 0.0381
  <br />  [19/20][600/1583]	Loss_D: 0.0710	Loss_G: 5.4901	D(x): 0.9791	D(G(z)): 0.0462 / 0.0081
  <br />  [19/20][650/1583]	Loss_D: 0.2505	Loss_G: 3.1630	D(x): 0.8928	D(G(z)): 0.1099 / 0.0610
  <br />  [19/20][700/1583]	Loss_D: 0.5833	Loss_G: 3.9856	D(x): 0.8569	D(G(z)): 0.2875 / 0.0325
  <br />  [19/20][750/1583]	Loss_D: 0.5008	Loss_G: 1.7952	D(x): 0.6639	D(G(z)): 0.0035 / 0.2825
  <br />  [19/20][800/1583]	Loss_D: 0.0785	Loss_G: 4.4216	D(x): 0.9554	D(G(z)): 0.0277 / 0.0247
  <br />  [19/20][850/1583]	Loss_D: 0.2170	Loss_G: 3.2202	D(x): 0.9020	D(G(z)): 0.0731 / 0.0593
  <br />  [19/20][900/1583]	Loss_D: 0.0965	Loss_G: 4.5541	D(x): 0.9478	D(G(z)): 0.0339 / 0.0232
  <br />  [19/20][950/1583]	Loss_D: 0.7587	Loss_G: 1.8880	D(x): 0.6176	D(G(z)): 0.1392 / 0.2118
  <br />  [19/20][1000/1583]	Loss_D: 0.0681	Loss_G: 5.0277	D(x): 0.9803	D(G(z)): 0.0441 / 0.0124
  <br />  [19/20][1050/1583]	Loss_D: 0.9239	Loss_G: 1.3154	D(x): 0.5490	D(G(z)): 0.1124 / 0.3449
  <br />  [19/20][1100/1583]	Loss_D: 0.2015	Loss_G: 5.1364	D(x): 0.9422	D(G(z)): 0.1122 / 0.0123
  <br />  [19/20][1150/1583]	Loss_D: 0.1000	Loss_G: 4.3750	D(x): 0.9435	D(G(z)): 0.0337 / 0.0244
  <br />  [19/20][1200/1583]	Loss_D: 0.0634	Loss_G: 4.4012	D(x): 0.9620	D(G(z)): 0.0228 / 0.0245
  <br />  [19/20][1250/1583]	Loss_D: 0.1163	Loss_G: 6.0967	D(x): 0.9893	D(G(z)): 0.0919 / 0.0042
  <br />  [19/20][1300/1583]	Loss_D: 1.8999	Loss_G: 7.2711	D(x): 0.9916	D(G(z)): 0.7757 / 0.0013
  <br />  [19/20][1350/1583]	Loss_D: 0.4245	Loss_G: 3.6919	D(x): 0.7111	D(G(z)): 0.0166 / 0.0532
  <br />  [19/20][1400/1583]	Loss_D: 0.0654	Loss_G: 4.9443	D(x): 0.9873	D(G(z)): 0.0478 / 0.0151
  <br />  [19/20][1450/1583]	Loss_D: 0.6750	Loss_G: 1.7718	D(x): 0.6017	D(G(z)): 0.0736 / 0.2655
  <br />  [19/20][1500/1583]	Loss_D: 0.3889	Loss_G: 3.4791	D(x): 0.7383	D(G(z)): 0.0078 / 0.0612
  <br />  [19/20][1550/1583]	Loss_D: 0.0771	Loss_G: 4.4849	D(x): 0.9462	D(G(z)): 0.0188 / 0.0183

</div>
</details>

```python
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```


![png](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/output_18_0.png?raw=true)



```python
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
```

![png](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/dcgan2.gif?raw=true)
![png](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/dcgan.gif?raw=true)

```python
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```

![png](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/output_19_2.png?raw=true)
![png](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/DCGAN/output_20_0.png?raw=true)



```python

```
