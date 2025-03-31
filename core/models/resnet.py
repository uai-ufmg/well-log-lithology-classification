import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, balanced_accuracy_score

from .model_template import Model


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(Model):
    def __init__(self, ResBlock, layer_list, cfg:dict, logs:list[str], seq_size:int, batch_size:int, lr:float, epochs:int, channels:int, num_classes:int, device:str, verbose:bool=True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.ResBlock = ResBlock
        
        self.cfg = cfg
        self.logs = logs
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.device = device
        self.verbose = verbose
        
        self.conv1 = nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        self.fc = nn.Linear(512*ResBlock.expansion, seq_size * num_classes)
        
        self.batch_norm2 = nn.BatchNorm1d(seq_size * num_classes, eps=1e-5, affine=True)
    
    def forward(self, x):
        x = x.permute(0, 2, 1).float()
        
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        #x = x.reshape(x.shape[0], -1)
        x = x.view(-1, 512*self.ResBlock.expansion)
        
        x = self.fc(x)
        x = self.batch_norm2(x)
        x = x.view(-1, self.num_classes)
        
        return None, x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)


def ResNet18(cfg:dict, logs:list[str], seq_size:int, batch_size:int, lr:float, epochs:int, channels:int, num_classes:int, device:str, verbose:bool=True):
    return ResNet(Bottleneck, [2,2,2,2], cfg, logs, seq_size, batch_size, lr, epochs, channels, num_classes, device, verbose)
        
def ResNet50(cfg:dict, logs:list[str], seq_size:int, batch_size:int, lr:float, epochs:int, channels:int, num_classes:int, device:str, verbose:bool=True):
    return ResNet(Bottleneck, [3,4,6,3], cfg, logs, seq_size, batch_size, lr, epochs, channels, num_classes, device, verbose)
    
def ResNet101(cfg:dict, logs:list[str], seq_size:int, batch_size:int, lr:float, epochs:int, channels:int, num_classes:int, device:str, verbose:bool=True):
    return ResNet(Bottleneck, [3,4,23,3], cfg, logs, seq_size, batch_size, lr, epochs, channels, num_classes, device, verbose)

def ResNet152(cfg:dict, logs:list[str], seq_size:int, batch_size:int, lr:float, epochs:int, channels:int, num_classes:int, device:str, verbose:bool=True):
    return ResNet(Bottleneck, [3,8,36,3], cfg, logs, seq_size, batch_size, lr, epochs, channels, num_classes, device, verbose)
