import torch
import torch.nn as nn

class TracMed(nn.Module):
    def __init__(self):
        pass 
    def forward(self,x):
        return x
if __name__=="__main__":
    model=TracMed()
    x=torch.randn(1,3,256,256)
    y=model(x)
    print(y.shape)