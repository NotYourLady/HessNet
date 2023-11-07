import torch
import torch.nn as nn

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn,
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1,
                 padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn)

    def forward(self, x):
        x = self.up(x)
        return x

    
class Unet3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=16,
                 act=nn.Sigmoid(), depth=4):
        super(Unet3d, self).__init__()
        
        self.depth = depth
        
        filters = [in_channels,]
        for i in range(depth+1):
            filters.append(channels * (2**i))
        
        self.Pools = nn.ModuleList([])
        self.Convs = nn.ModuleList([])
        self.UpConvs = nn.ModuleList([])
        self.Ups = nn.ModuleList([])

        for i in range(1, depth+1):
            self.Pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.Convs.append(conv_block(filters[i], filters[i+1]))
            self.UpConvs.append(conv_block(filters[i+1], filters[i]))
            self.Ups.append(up_conv(filters[i+1], filters[i]))

        self.inConv = conv_block(in_channels, channels)
        self.outConv = nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.act = act
        
    def forward(self, x):
        down_features = []    
        down_features.append(self.inConv(x))
        
        for i in range(self.depth):
            down_features.append(self.Convs[i](self.Pools[i](down_features[i])))

        for i in reversed(range(self.depth)):
            x = self.Ups[i](down_features[i+1])
            x = torch.cat((down_features[i], x), dim=1)
            x = self.UpConvs[i](x)

        out = self.outConv(x)     
        out = self.act(out)
        return out
    
    
    
    
class U_Net(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, channels=16, act=nn.Sigmoid()):
        super(U_Net, self).__init__()
        
        n1 = channels #
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 64,128,256,512,1024

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.act = act

    def forward(self, x):    
        e1 = self.Conv1(x)
        
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        out = self.Conv(d2)
        out = self.act(out)
        return out
    
    
def get_model(model_name):
    if model_name == 'Unet3d_16ch':
        #return(Unet3d(channels=16, depth=4))
        return(U_Net(channels=16))
    else:
        return None