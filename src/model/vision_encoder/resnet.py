import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_subtask_classes=4,num_finaltask_classes=3,num_classes=3, n_subtasks=3,n_finaltasks=1,device="cuda"):
        super(ResNet3D, self).__init__()
        self.device=device
        # self.embed_dim=512
        self.embed_dim=32
        expansion=64
        self.in_planes = 32
        self.n_subtasks=n_subtasks
        self.n_finaltasks=n_finaltasks
        self.num_subtask_classes=num_subtask_classes
        self.num_finaltask_classes=num_finaltask_classes
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear_subtask=nn.Linear(256*expansion,self.embed_dim*n_subtasks)
        self.linear_finaltask=nn.Linear(256*expansion,self.embed_dim*n_finaltasks)
        self.cls_subtask=nn.Linear(self.embed_dim,num_subtask_classes)
        self.cls_finaltask=nn.Linear(self.embed_dim,num_finaltask_classes)
        
        # self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("out",out.shape)
        out = self.layer1(out)
        # print("out",out.shape)
        out = self.layer2(out)
        # print("out",out.shape)
        out = self.layer3(out)
        # print("out",out.shape)
        
        out = self.layer4(out)
        # print("out",out.shape)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        subtask_out = self.linear_subtask(out)
        finaltask_out = self.linear_finaltask(out)
        # print("subtask_out",subtask_out.shape,"finaltask_out",finaltask_out.shape)
        subtask_out=subtask_out.view(out.size(0),self.n_subtasks,-1)
        finaltask_out=finaltask_out.view(out.size(0),self.n_finaltasks,-1)
        subtask_out=self.cls_subtask(subtask_out)
        finaltask_out=self.cls_finaltask(finaltask_out)
        # subtask_out=self.softmax(subtask_out)
        # finaltask_out=self.softmax(finaltask_out)
        return subtask_out,finaltask_out

def ResNet18_3D():
    return ResNet3D(BasicBlock3D, [2,2,2,2])

if __name__ == "__main__":
    model = ResNet18_3D()
    print("model",model)
    model.to("cuda")
    inp = torch.randn(4,1,32,256,256)
    inp = inp.to("cuda")
    subtask_out,finaltask_out = model(inp)
    print(subtask_out.shape,finaltask_out.shape)