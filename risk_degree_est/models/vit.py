import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch import nn

from einops.layers.torch import Rearrange

from einops import repeat
from einops.layers.torch import Rearrange

import numpy as np

class Patching(nn.Module):
    
    """
    入力
        torch.Size([100, 3, 32, 32])
    出力
        torch.Size([100, 64, 48])
    """
    
    def __init__(self, x_patch_size, y_patch_size):
        """ [input]
            - patch_size (int) : パッチの縦の長さ（=横の長さ）
        """
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = y_patch_size, pw = x_patch_size)
    
    def forward(self, x):
        """ [input]
            - x (torch.Tensor) : 画像データ
                - x.shape = torch.Size([batch_size, channels, image_height, image_width])
        """
        x = self.net(x)
        return x


class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):#dim = 256
        """ [input]
            - patch_dim (int) : 一枚あたりのパッチのベクトルの長さ（= channels * (patch_size ** 2)）
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ 
        """
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        """ [input]
            - x (torch.Tensor) 
                - x.shape = torch.Size([batch_size, n_patches, patch_dim])
        """
        x = self.net(x)
        return x

    
class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_patches (int) : パッチの枚数
        """
        super().__init__()
        # [class] token
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))#dim = 256
        self.region_linear = torch.nn.Linear(2, dim)
        # position embedding
        self.img_embeding = nn.Parameter(torch.randn(1, n_patches-1, dim))

    
    def forward(self, x, center_prop):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches, dim])
        """
        center_prop = center_prop.to("cuda")
        #print(x.shape)
        #print(self.pos_embedding.shape)
        # バッチサイズを抽出
        #batch_size, _, __ = x.shape#batch_size 100
        
        region_embeding = self.region_linear(center_prop)
        # [class] トークン付加
        region_embeding = torch.unsqueeze(region_embeding,dim=0)
        # x.shape : [batch_size, n_patches, patch_dim] -> [batch_size, n_patches + 1, patch_dim]
        #cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size).cuda()#バッチ数分同じ配列を重ねる
        #print(x.shape,"x")#torch.Size([100, 64, 256])
        #print(region_embeding.shape)
        #rint(self.img_embeding.shape)
        #print(cls_tokens.shape,"token")#torch.Size([100, 1, 256])
        pos_embedding = torch.cat([region_embeding, self.img_embeding], dim = 1)
        #x = torch.cat([cls_tokens, x], dim = 1)
        
        #print(pos_embedding.shape)#torch.Size([100, 65, 256])
        #x = x.to("cuda")
        #print(self.pos_embedding.shape)#torch.Size([1, 65, 256])
        # 位置エンコーディング
        x += pos_embedding

        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - hidden_dim (int) : 隠れ層のノード数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        x = self.net(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_heads (int) : heads の数
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h = self.n_heads)

        self.softmax = nn.Softmax(dim = -1)

        self.cat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        #print(x.shape)# torch.Size([100, 65, 256])
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        #print(q.shape)# q torch.Size([100, 65, 256])

        q = self.split_into_heads(q)#ヘッド数分、分割
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # Logit[i] = Q[i] * tK[i] / sqrt(D) (i = 1, ... , n_heads)
        # AttentionWeight[i] = Softmax(Logit[i]) (i = 1, ... , n_heads)
        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_heads ** -0.5)
        attention_weight = self.softmax(logit)

        # Head[i] = AttentionWeight[i] * V[i] (i = 1, ... , n_heads)
        # Output = concat[Head[1], ... , Head[n_heads]]
        output = torch.matmul(attention_weight, v)
        output = self.cat(output)
        #print(output.shape)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        """ [input]
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - mlp_dim (int) : MLP の隠れ層のノード数
        """
        super().__init__()

        # Layers
        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim = dim, n_heads = n_heads)
        self.mlp = MLP(dim = dim, hidden_dim = mlp_dim)
        self.depth = depth#　3　なので4層

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """

        
        for _ in range(self.depth):
            
            x = self.multi_head_attention(self.norm(x)) + x# 入力x (torch.Size([100, 65, 256]))　出力x (torch.Size([100, 65, 256]))
            #print(x.shape)
            x = self.mlp(self.norm(x)) + x

        return x


class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    
class ViT(nn.Module):
    def __init__(self, width_image_size, height_image_size, width_patch_size, height_patch_size, dim, depth, n_heads, channels, mlp_dim, region_feat_dim):
        
        super().__init__()
        
        # Params
        n_patches = (width_image_size // width_patch_size) * (height_image_size // height_patch_size)
        patch_dim = channels * width_patch_size * height_patch_size#1パッチのユニット数
        self.depth = depth

        # Layers
        self.patching = Patching(width_patch_size, height_patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim)
        #self.region_linear = Region_Linear(input_dim = 256 * 7 * 7, dim = dim)
        self.embedding = Embedding(dim = dim, n_patches = n_patches + 1)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth)
        self.mlp_head = MLPHead(dim = dim, out_dim = 2048)

    def box_center(self, proposal, image_shape):
        center_box_list = []
        print(proposal.shape)
        for one_box in proposal:
            x = ((one_box[2] - one_box[0]) / 2 + one_box[0]) / image_shape[0][1]
            y = ((one_box[3] - one_box[1]) / 2 + one_box[1]) / image_shape[0][0]
            center_box = torch.tensor([x,y])
            center_box_list.append(center_box)
            #print(center_box)
        center_box_tensor = torch.stack(center_box_list)
        return center_box_tensor
    
    def forward(self, img, region, proposal, image_shapes):
        """ 
        img size torch.Size([1, 256, 50, 68])
        region size torch.Size([512, 256, 7, 7])
        """
        prop = proposal[0]
        center_prop = self.box_center(prop, image_shapes)
        
        
        x = img#torch.Size([1, 256, 50, 68])
        
        x = self.patching(x)#torch.Size([1, 20, 43520])
        
        x = self.linear_projection_of_flattened_patches(x)#torch.Size([1, 20, 4096])
        
        #print(x.shape,"x")
        #print(region.shape)
        region = region.unsqueeze(dim=0)#torch.Size([1, 512, 4096])
        #print(region.shape)
        #print(x.shape)
        x = torch.cat([region , x], dim = 1)#torch.Size([1, 532, 4096])
        #print(x.shape)
        x = self.embedding(x, center_prop)#torch.Size([1, 532, 4096])
        #print(x.shape)
        x = self.transformer_encoder(x)#torch.Size([1, 22, 4096])
        #print(x.shape)
        
        #print(x.shape)
        """
        for i in range(len(region[0])):
            #print(x.shape)
            region_x = x[:, i]
            #print(region_x.shape)#torch.Size([1, 4096])
            one_x = region_x.squeeze()
            #print(one_x.shape)#torch.Size([4096])
            one_x_out = self.mlp_head(one_x)
            #print(one_x_out[1:10])
            new_region.append(one_x_out)
        
       """
        #print(x.shape)
        x = x[0][:512]
 
        x_out = self.mlp_head(x)
        

        #print(x_out.shape)
        
        return x_out