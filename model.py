import torch
import torch.nn as nn
import Pyramid
import PatchEmbedding
import PositionEmbedding
import FeedForward
import Transformer
import SelectiveKernelConv
import FulllyConnectedLayer
import DepthwiseSeparableConvolution
class MNT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding=PatchEmbedding.PatchEmbedding()
        self.position_embedding=PositionEmbedding.PositionEmbedding()
        self.feedforward=FeedForward.FeedForward()
        self.attention=Transformer.Attention()
        self.FullyconnectedLayer=FulllyConnectedLayer.FullyConnected()
        self.DSC=DepthwiseSeparableConvolution.DepthwiseSeparableConvolution()

    def forward(self, input):
        x=self.patch_embedding(input)
        x=self.position_embedding(x)
        identity=x
        x=self.feedforward(x)
        x=self.attention(x)
        x+=identity
        identity=x
        x=self.feedforward(x)
        x+=identity

class pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramid=Pyramid.Pyramid()

    def forward(self, x):
        x=self.pyramid(x)
        return x

class ADD(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcl=FulllyConnectedLayer.FullyConnected()
        self.skc=SelectiveKernelConv.SKUnit()

    def forward(self, x):
        x=self.fcl(x)
        x=self.skc(x)
        return x


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnt=MNT()
        self.pyramid=pyramid()
        self.ADD=ADD()

    def forward(self, x):
        mnt_outputs=[]
        for i in range(5):
            x=self.mnt(x)
            mnt_outputs.append(x)
        x=self.pyramid(x)
        x-=self.ADD(x)

        return x,mnt_outputs
