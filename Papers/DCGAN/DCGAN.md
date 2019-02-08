

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
    




    <torch._C.Generator at 0x1b1f6ddaa30>



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
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
```

ImageFolder


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




    <matplotlib.image.AxesImage at 0x1b1e036eda0>




![png](output_4_1.png)


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
    


```python
device
```




    device(type='cuda', index=0)



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
import torch
a = torch.full((64, 100, 1, 1), 1)
```


```python
a.mean().item()
```




    1.0




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

    Starting Training Loop...
    [0/5][0/1583]	Loss_D: 1.7410	Loss_G: 4.7765	D(x): 0.5343	D(G(z)): 0.5771 / 0.0136
    


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


```python
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
```


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
