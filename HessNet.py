import torch
import torch.nn as nn
from itertools import combinations_with_replacement


def get_gauss_kernel_torch(sigma):
    sigma = torch.tensor([0.1, torch.tensor([sigma, 10]).min()]).max() # 0.1 < sigma < 10
    kernel_size = 1+6*int(sigma)
    kernel = torch.arange(kernel_size).type(torch.float32) - kernel_size//2
    kernel = (kernel)**2 / 2 / sigma**2
    kernel = torch.exp(-kernel)
    return kernel/kernel.sum()


class GaussianBlur3D(nn.Module):
    def __init__(self, sigma):
        super(GaussianBlur3D, self).__init__()
        kernel_size = 1+6*int(sigma)
        gaussian_kernel = get_gauss_kernel_torch(sigma)
        
        kernel3d = torch.einsum('i,j,k->ijk', gaussian_kernel, gaussian_kernel, gaussian_kernel)
        kernel3d /= kernel3d.sum()
        
        self.conv = nn.Conv3d(1, 1, kernel_size, stride=1, padding=kernel_size//2,
                              dilation=1, bias=False, padding_mode='replicate')
        self.conv.weight = torch.nn.Parameter(kernel3d.reshape(1, 1, *kernel3d.shape), requires_grad=False)
        
    def forward(self, vol):
        return self.conv(vol)


class GaussianBlur3D(nn.Module):
    def __init__(self, sigma):
        super(GaussianBlur3D, self).__init__()
        self.kernel_size = 1+6*int(sigma)
        
        self.conv = nn.Conv3d(1, 1, self.kernel_size, stride=1, padding=self.kernel_size//2,
                              dilation=1, bias=False, padding_mode='replicate')
        self.set_weights(sigma)
    
    def set_weights(self, sigma):
        sigma = torch.tensor([0.1, torch.tensor([sigma, 10]).min()]).max() # 0.1 < sigma < 10
        kernel1d = torch.arange(self.kernel_size).type(torch.float32) - self.kernel_size//2
        kernel1d = (kernel1d)**2 / 2 / sigma**2
        kernel1d = torch.exp(-kernel1d)
        kernel3d = torch.einsum('i,j,k->ijk', kernel1d, kernel1d, kernel1d)
        kernel3d /= kernel3d.sum()
        self.conv.weight = torch.nn.Parameter(kernel3d.reshape(1, 1, *kernel3d.shape), requires_grad=False)
        
    def forward(self, vol, sigma=None):
        if sigma is not None:
            self.set_weights(sigma)
        return self.conv(vol)


class HessianTorch(nn.Module):
    def __init__(self, sigma):
        super(HessianTorch, self).__init__()
        self.gauss = GaussianBlur3D(sigma=sigma)
        
    def forward(self, vol):
        axes = [2, 3, 4]
        gaussian_filtered = self.gauss(vol)
        
        gradient = torch.gradient(gaussian_filtered, dim=axes)
        H_elems = [torch.gradient(gradient[ax0-2], axis=ax1)[0]
              for ax0, ax1 in combinations_with_replacement(axes, 2)]
        return torch.stack(H_elems)


def nn_detect(vol, scale, l_func, device='cpu'):
    H = HessianTorch(scale)(vol).permute(1,2,3,4,5,0)
    H = torch.flatten(H, start_dim=1, end_dim=4)
    scale_attention = scale*torch.ones((H.shape[0], H.shape[1], 1)).to(device)
    x = torch.cat([H, scale_attention], axis=2)
    x = l_func(x)
    x = torch.unflatten(x, 1, vol.shape[2:])
    x = x.permute(0,4,1,2,3)
    return x[0, 0].detach().numpy()    

    
class HessBlock(nn.Module):
    def __init__(self, start_scale, patch_size, device): #start scale - experimentaly
        super(HessBlock, self).__init__()
        
        self.device = device
        self.scale = nn.parameter.Parameter(data=torch.tensor(start_scale, dtype=torch.float32))
        
        self.linear = nn.Sequential(
            nn.Linear(7, 10, bias=True),
            nn.ReLU(),
            #nn.Linear(10, 10, bias=True),
            #nn.ReLU(),
            nn.Linear(10, 1, bias=True),
            nn.Sigmoid()
        )
        self.hess = HessianTorch(self.scale.item())
        self.flat = nn.Flatten(start_dim=1, end_dim=4)
        self.unflat = torch.nn.Unflatten(1, patch_size)
        
        
    def forward(self, x):
        x = self.hess(x).permute(1,2,3,4,5,0) #1
        x = self.flat(x)
        scale_attention = self.scale*torch.ones((x.shape[0], x.shape[1], 1)).to(self.device)
        x = torch.cat([x, scale_attention], axis=2)
        x = self.linear(x) #2
        x = self.unflat(x)
        x = x.permute(0,4,1,2,3)
        return x
    
    
