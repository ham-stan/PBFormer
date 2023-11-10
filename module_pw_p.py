import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # bnc
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)  # 3b h n c/h
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)         b h n c/h

        attn = (q @ k.transpose(-2, -1)) * self.scale  # b h n n
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)  # bhn c/h -> bnc
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, attn_drop=0.2,
                 drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):  # bnc
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x  # bnc


class TransformerEncoder(nn.Module):
    """
    ViT-B:
        depth=12, head=12;
    ViT-L:
        depth=24, head=16;
    ViT-H:
        depth=32, head=16;
    """
    def __init__(self, depth=4, embed_dim=384, num_head=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0.2,
                 attn_drop_rate=0.2, drop_path_rate=0.2, output_slice=True):
        super(TransformerEncoder, self).__init__()
        self.slice = output_slice
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_head, mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                           attn_drop=attn_drop_rate,
                                           drop_path=drop_path_rate[i]
                                           if isinstance(drop_path_rate, list) else drop_path_rate)
                                     for i in range(depth)])

    def forward(self, final_input):
        tmp = final_input
        for block in self.blocks:
            tmp = block(tmp)
        if self.slice:
            cls_token = tmp[:, 0, :].unsqueeze(1)  # b 1 c
            representation = tmp[:, 1:, :]  # b seq_len c
            return cls_token, representation
        else:
            return tmp  # b seq_len+1 c


class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.2, attn_drop=0.2, drop_path=0.2,
                 norm_layer=nn.LayerNorm):
        super(CrossBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):  # b n+1 c
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x


class BTSCrossEncoder(nn.Module):  # time space cross    bi-temporal-spatial BTS
    def __init__(self, depth=4, embed_dim=384, num_head=8, qkv_bias=False, qk_scale=None, drop_rate=0.2,
                 attn_drop_rate=0.2, drop_path_rate=0.2, mode='org'):
        super(BTSCrossEncoder, self).__init__()
        self.mode = mode
        self.blocks = nn.ModuleList([CrossBlock(dim=embed_dim, num_heads=num_head,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=drop_path_rate[i]
                                                if isinstance(drop_path_rate, list) else drop_path_rate)
                                     for i in range(depth)])
        self.proj = nn.Linear(embed_dim*2, embed_dim)

    def forward(self, x1, cls1, x2, cls2):  # bnc b1c bnc b1c
        tmp1 = torch.cat((cls1, x2), dim=1)
        tmp2 = torch.cat((cls2, x1), dim=1)
        for idx, cb in enumerate(self.blocks):
            cls1 = cb(tmp1)[:, 0].unsqueeze(1)
            cls2 = cb(tmp2)[:, 0].unsqueeze(1)
            if idx % 2 == 0:
                tmp1 = torch.cat((cls1, x1), dim=1)
                tmp2 = torch.cat((cls2, x2), dim=1)
            else:
                tmp1 = torch.cat((cls1, x2), dim=1)
                tmp2 = torch.cat((cls2, x1), dim=1)
        if self.mode == 'cat':
            cls = self.proj(torch.cat((cls1, cls2), dim=-1))
        elif self.mode == 'sub':
            cls = cls1 - cls2
        else:  # org
            cls = cls1 + cls2
        return cls


class Embedding(nn.Module):
    def __init__(self, embedding_dim=384, seq_l=256, pos=True, pos_self=True):
        super(Embedding, self).__init__()
        self.pos = pos
        self.pos_self = pos_self
        self.patch_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, embedding_dim, 1),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embedding_dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_l, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embedding_dim))
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def forward(self, x):
        b, _, seq_len, c = x.shape  # b2n3
        pt_1, pt_2 = x[:, 0, :, :].squeeze(), x[:, 1, :, :].squeeze()  # bn3, bn3
        pt1 = self.patch_embed(pt_1.transpose(2, 1)).transpose(2, 1)  # bnc(512)
        pt2 = self.patch_embed(pt_2.transpose(2, 1)).transpose(2, 1)  # bnc(512)
        cls_token = self.cls_token.expand(b, -1, -1)
        pt1, pt2 = torch.cat((pt1, cls_token), dim=1), torch.cat((pt2, cls_token), dim=1)
        if self.pos:
            if self.pos_self:
                pos1, pos2 = self.pos_embed(pt_1), self.pos_embed(pt_2)
            else:
                pos1, pos2 = self.pos_embedding.expand(b, -1, -1), self.pos_embedding.expand(b, -1, -1)
            cls_pos = self.cls_pos.expand(b, -1, -1)
            pos1, pos2 = torch.cat((pos1, cls_pos), dim=1), torch.cat((pos2, cls_pos), dim=1)
            return pt1+pos1, pt2+pos2  # (b n+1(513) 512) (b n+1(513) 512)
        else:
            return pt1, pt2  # (b n+1(513) 512) (b n+1(513) 512)


class PNEmbedding(nn.Module):
    def __init__(self, embed_dim=384, pos=True, seq_len=256):
        super(PNEmbedding, self).__init__()
        self.pos = pos
        self.pn1 = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        self.pn2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        trunc_normal_(self.pos_embedding, std=.02)
        self.feature_seq = nn.Linear(1, seq_len)

    def forward(self, x):  # 1 n c
        _, n, c = x.shape
        # x = x.transpose(2, 1)  # 1 c n
        x = self.pn1(x)  # 1 n c
        g = torch.max(x, dim=1, keepdim=True)[0]  # 1 1 c
        x = torch.cat((x, g.expand(-1, n, -1)), dim=-1)  # 1 n 2c
        # print('-------', x.shape)
        x = self.pn2(x)  # 1 n embed_dim
        if self.pos:
            pos = self.pos_embedding.expand(-1, n, -1)
            x = x + pos
        x = torch.max(x, dim=1, keepdim=True)[0]  # 1 1 embed_dim
        x = self.feature_seq(x.transpose(2, 1)).transpose(2, 1)
        return x  # 1 seq_len embed_dim


class NaiveCross(nn.Module):
    def __init__(self, depth=4, embed_dim=384, num_head=8, qkv_bias=False, qk_scale=None, drop_rate=0.2,
                 attn_drop_rate=0.2, drop_path_rate=0.2, seq_len=256):
        super(NaiveCross, self).__init__()
        self.blocks = nn.ModuleList([CrossBlock(dim=embed_dim, num_heads=num_head,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=drop_path_rate[i]
                                                if isinstance(drop_path_rate, list) else drop_path_rate)
                                     for i in range(depth)])
        self.proj = nn.Conv1d(seq_len * 2, 1, 1)

    def forward(self, x):  # 1 2n c
        for _, bk in enumerate(self.blocks):
            x = bk(x)  # 1 2n c
        x = self.proj(x)  # 1 1 c
        return x
