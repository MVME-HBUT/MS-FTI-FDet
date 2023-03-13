import torch
import torch.nn as nn
#import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

def init_conv_weights(layer, weights_std=0.01,  bias=0):
    '''
    RetinaNet's layer initialization
    :layer
    :
    '''
    nn.init.normal_(layer.weight, std=weights_std)
    nn.init.constant_(layer.bias, val=bias)
    return layer


def conv1x1(in_channels, out_channels, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)

    return layer


def conv3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)

    return layer

def zy_1(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Sequential(conv1x1(in_channels, out_channels),conv1x1(out_channels, out_channels))

    return layer



def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

#######################################################################
#@定义SE注意力模块
class SELayer(nn.Module):
    def __init__(self, in_channels,channel, reduction=16):
        super(SELayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channel, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

######################################################################

class Bottleneck(nn.Module):
    # 这里对应是4,对应每层中的64，64，256
      
    def __init__(self,in_channel,out_channel,stride,expansion):
        super(Bottleneck,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                             kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                             kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        
        self.conv3=nn.Conv2d(in_channels=out_channel,out_channels=out_channel*expansion,
                             kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channel*expansion)
        
        self.relu=nn.ReLU(inplace=True)
        
            
    def forward(self,x):
        identity=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        
        out=self.conv3(out)
        out=self.bn3(out)
        
        out+=identity
        #out=self.relu(out)
        
        return out
 
##这个没有shoutcut路径
class Bottleneck_2(nn.Module):
    # 这里对应是4,对应每层中的64，64，256
      
    def __init__(self,in_channel,out_channel,expansion,stride=2):
        super(Bottleneck_2,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                             kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                             kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        
        self.conv3=nn.Conv2d(in_channels=out_channel,out_channels=out_channel*expansion,
                             kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channel*expansion)
        
        self.relu=nn.ReLU(inplace=True)

        self.shortcut = nn.Conv2d(
                in_channel,
                out_channel*expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        
            
    def forward(self,x):
           
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        
        out=self.conv3(out)
        out=self.bn3(out)

        shortcut = self.shortcut(x)
        out += shortcut

        #out=self.relu(out)
        
        return out

##这个没有shoutcut路径
class Bottleneck_3(nn.Module):
    # 这里对应是4,对应每层中的64，64，256
    #!注意这里的expansion，为了修改通道数改为了1
    # def __init__(self,in_channel,out_channel,stride,expansion=4):
    def __init__(self,in_channel,out_channel,stride,expansion=2):
        super(Bottleneck_3,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                             kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                             kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        
        self.conv3=nn.Conv2d(in_channels=out_channel,out_channels=out_channel*expansion,
                             kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channel*expansion)
        
        self.relu=nn.ReLU(inplace=True)

        self.shortcut = nn.Conv2d(
                in_channel,
                out_channel*expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        
            
    def forward(self,x):
        #identity=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        
        out=self.conv3(out)
        out=self.bn3(out)
        
        shortcut = self.shortcut(x)
        out += shortcut
        
        #out=self.relu(out)
        
        return out