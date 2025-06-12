from torch import nn
# from .spatial_pooling_projector import SpatialPoolingProjector
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class Minigpt(nn.Module):
    def __init__(self, config=None):
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x




class FullLinear(nn.Module):
    def __init__(self, config):
        super(FullLinear, self).__init__()
        self.linear = nn.Linear(config.mm_hidden_size, config.hidden_size)
    def forward(self, x):
        x = self.linear(x)
        return x
    @property
    def proj_out_num(self):
        num = 2048
        return num


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type')
    print(" build projector",projector_type)
    if projector_type == 'linear':
        return FullLinear(config)
    # ([2, 2048, 768])
    # in_dim=config.mm_hidden_size= 512
    #
    elif projector_type == 'spp':
        return SpatialPoolingProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_type=config.proj_pooling_type,
                                        pooling_size=config.proj_pooling_size)


    elif projector_type == 'identity':
        return IdentityMap()
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')




class SpatialPoolingProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
            
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        if layer_type == 'linear':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            print("Projector error!")

        self.pooling_type = pooling_type

    def forward(self, x):
        print("x shape",x.shape)
        B = x.shape[0] # B*N*D
        print("self.num_patches_pre",self.num_patches_pre)
        if self.pooling_type == 'spatial':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x)
            x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x)
        elif self.pooling_type == 'sequence':
            x = x.permute(0, 2, 1) #b d n
            x = F.avg_pool1d(x, kernel_size=self.pooling_size**3, stride=self.pooling_size**3)
            x = x.permute(0, 2, 1) #b n d

        x = rearrange(x, "b n d -> (b n) d")
        x = self.projector(x)
        x = rearrange(x, "(b n) d -> b n d", b=B)

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num