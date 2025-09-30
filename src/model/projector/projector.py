from torch import nn
from torch import nn
import torch.nn.functional as F
import os 
from dotenv  import load_dotenv
load_dotenv()
import sys
ROOT=os.getenv("ROOT")
sys.path.append(ROOT)

from einops import rearrange
from einops.layers.torch import Rearrange

import yaml
# from utils.type import dict_to_namespace

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class Minigpt(nn.Module):
    def __init__(self, config):
        
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        in_channels,out_channels,in_embed,out_embed = config["in_channels"], config["out_channels"], config["in_embed"], config["out_embed"] 
        # inc, ouc = in_dim, out_dim
        # self.linear = nn.Linear(inc * 4, ouc)
        if in_channels % out_channels != 0:
            raise ValueError(f"in_channels {in_channels} must be divisible by out_channels {out_channels}")
        self.factor = in_channels // out_channels
        # self.factor=in_channels/
        self.linear = nn.Linear(in_embed * self.factor, out_embed)
    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        # if num_tokens % 4 != 0:
        #     raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // self.factor, c * self.factor)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config["mm_hidden_size"], config["hidden_size"]
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
    def __init__(self, in_dim,hidden_dim, out_dim):
        super(FullLinear, self).__init__()
        self.projector=nn.Sequential(
              nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim)   #
            #    nn.Linear(in_dim, out_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            # nn.Linear(out_dim,out_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            )
        self._init_weights()
    def _init_weights(self):
        for layer in self.projector.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)  # Small positive bias
                print(f"Initialized {layer} with Xavier weights and bias=0.1")
    def forward(self, x):
        x = self.projector(x)
        return x
    @property
    def proj_out_num(self):
        num = 2048
        return num

def load_mm_projector(config):
    """
    Load the multimodal projector based on the configuration.
    
    Args:
        config (dict): Configuration dictionary containing projector type and parameters.
        
    Returns:
        nn.Module: The initialized multimodal projector.
    """
    if config["mm_projector_type"] == 'linear':
        return FullLinear(in_dim=config["in_embed"], hidden_dim=config["hidden_dim"], out_dim=config["out_embed"])
    elif config["mm_projector_type"] == 'spp':
        print("using spatial pooling projector")
        return SpatialPoolingProjector(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            in_dim=config["in_embed"],
            out_dim=config["out_embed"],
            layer_type=config["layer_type"],
            layer_num=config["layer_num"],
            pooling_type=config["pooling_type"],
            pooling_size=config["pooling_size"]
        )
    elif config["mm_projector_type"] == 'minigpt':
        return Minigpt(config)
    elif config["mm_projector_type"] == 'identity':
        return IdentityMap()
    elif config["mm_projector_type"] == 'transpose':
        return TransposeProjector(config)
    else:
        raise ValueError(f'Unknown projector type: {config.mm_projector_type}')




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
            self.projector = nn.Sequential(modules)
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
        B = x.shape[0] # B*N*D
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


class TransposeProjector(nn.Module):
    def __init__(self,config, vit_dim=256, output_channels=1, patches=(8, 16, 16)):
        super().__init__()
        self.patches = patches
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(vit_dim, 256, kernel_size=2, stride=2),  # 2x up
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),  # 4x up
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.out_conv = nn.Conv3d(128, output_channels, kernel_size=1)  # Channel adjust

    def forward(self, x):
        # Input: (B, N, vit_dim) → reshape to 3D
        B, N, C = x.shape
        D, H, W = self.patches
        x = x.permute(0, 2, 1).view(B, C, D, H, W)  # (B, C, D, H, W)
        x = self.up1(x)  # 2x up
        x = self.up2(x)  # 4x up
        return self.out_conv(x)  # (B, output_channels, D*4, H*4, W*4)


class UNetProjector(nn.Module):
    def __init__(self, vit_dim, vit_layers):
        super().__init__()
        # ViT intermediate features (from specified layers)
        self.skip_conns = vit_layers  
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(vit_dim, 256, 2, stride=2),
                nn.BatchNorm3d(256),
                nn.ReLU()
            ) for _ in range(len(vit_layers))
        ])
        self.fuse = nn.Conv3d(256 * len(vit_layers), 256, kernel_size=1)

    def forward(self, x, vit_intermediates):
        # x: main ViT output; vit_intermediates: list of skip features
        features = []
        for i, (block, skip) in enumerate(zip(self.decoder_blocks, vit_intermediates)):
            x = block(x)
            x = x + skip  # Skip connection
            features.append(x)
        return self.fuse(torch.cat(features, dim=1))
if __name__ == "__main__":
    import torch
    # config = SimpleNamespace(mm_hidden_size=2560, hidden_size=758)

    projector = build_mm_projector()
    x = torch.randn(2, 2048, 2560)  # Example input
    output = projector(x)
    print("Output shape:", output.shape)  # Should be [2, 2048, 758] if spp is used