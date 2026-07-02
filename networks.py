import torch
import math

class Convolution(torch.nn.Module):

    def __init__(self,layer):
        super().__init__()

        layers=[]
        for i in range(len(layer)-1):
            layers.append(torch.nn.Conv2d(layer[i],layer[i+1],3,1,1))
            if(i+1<len(layer)-1):
                layers.append(torch.nn.LeakyReLU(0.2))

        self.network = torch.nn.Sequential(*layers)

    def forward(self,x):
        y = self.network(x)
        return y
    
class DenseBlock(torch.nn.Module):

    def __init__(self,features,growth_rate,no_blocks):
        super().__init__()

        self.conv = torch.nn.ModuleList()
        for i in range(no_blocks):
            if(i<no_blocks-1):
                self.conv.append(torch.nn.Conv2d(features+growth_rate*i,growth_rate,3,1,1))
            else:
                self.conv.append(torch.nn.Conv2d(features+growth_rate*i,features,3,1,1))
        
        self.lrelu = torch.nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
        z = x
        for i in range(len(self.conv)):
            y = self.conv[i](z)
            if(i<len(self.conv)-1):
                y = self.lrelu(y)
                z = torch.cat([z,y],dim=1)
        
        return x + y*0.2
    
class RRDB(torch.nn.Module):

    def __init__(self,features,growth_rate,no_blocks):
        super().__init__()

        self.db1 = DenseBlock(features,growth_rate,no_blocks)
        self.db2 = DenseBlock(features,growth_rate,no_blocks)
        self.db3 = DenseBlock(features,growth_rate,no_blocks)
    
    def forward(self,x):
        y1 = self.db1(x)
        y2 = self.db2(y1)
        y3 = self.db3(y2)

        return x + 0.2*y3

class RRDBs(torch.nn.Module):

    def __init__(self,no_residual_block,features,growth_rate,no_blocks):
        super().__init__()

        self.network = torch.nn.Sequential(*[RRDB(features,growth_rate,no_blocks) for i in range(no_residual_block)])
    
    def forward(self,x):
        return self.network(x)
    
class Upsample(torch.nn.Module):

    def __init__(self,scale,feature_size):
        super().__init__()
        
        layers = []
        for i in range(int(math.log2(scale))):
            layers.append(torch.nn.Conv2d(feature_size,4*feature_size,3,1,1))
            layers.append(torch.nn.PixelShuffle(2))
            layers.append(torch.nn.LeakyReLU(0.2,inplace=True))
        
        layers.append(torch.nn.Conv2d(feature_size,feature_size,3,1,1))
        layers.append(torch.nn.LeakyReLU(0.2,inplace=True))
        layers.append(torch.nn.Conv2d(feature_size,3,3,1,1))

        self.network = torch.nn.Sequential(*layers)
    
    def forward(self,feature_maps):
        return self.network(feature_maps)
    
class Generator(torch.nn.Module):

    def __init__(self,out_channel,no_residual_block,growth_rate,no_blocks,scale):     # 64, 23, 32, 5, 4
        super().__init__()

        self.initial_conv = torch.nn.Conv2d(3,out_channel,3,1,1)
        self.rrdbs = RRDBs(no_residual_block,out_channel,growth_rate,no_blocks)
        self.trunk_conv = torch.nn.Conv2d(out_channel,out_channel,3,1,1)
        self.upsample = Upsample(scale,out_channel)

    def forward(self,x):
        first_conv = self.initial_conv(x)
        dense_features = self.rrdbs(first_conv)
        refined_features = self.trunk_conv(dense_features)
        special_features = first_conv + refined_features

        result = self.upsample(special_features)

        return result
    
class Discriminator(torch.nn.Module): # for 256 X 256 X 3

    def __init__(self):
        super().__init__()

        channels = [3, 64, 128, 256, 512]
        layers = []
        for i in range(len(channels)-1):
            layers.append(torch.nn.utils.spectral_norm( torch.nn.Conv2d(channels[i],channels[i+1],3,1,1)))
            layers.append(torch.nn.LeakyReLU(0.2,inplace=True))
            layers.append(torch.nn.utils.spectral_norm( torch.nn.Conv2d(channels[i+1],channels[i+1],3,2,1)))
            layers.append(torch.nn.LeakyReLU(0.2,inplace=True))

        layers.append(torch.nn.utils.spectral_norm( torch.nn.Conv2d(512,1,3,1,1)))
        self.conv_network = torch.nn.Sequential(*layers)

    def forward(self,x):
        y = self.conv_network(x)
        return y
    
