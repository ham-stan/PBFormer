import torch
import torch.nn as nn
from module_pw_p import Embedding, TransformerEncoder, BTSCrossEncoder, PNEmbedding, NaiveCross


class PCDTransformer(nn.Module):
    def __init__(self, ):
        super(PCDTransformer, self).__init__()
        self.embedding = Embedding(embedding_dim=384, seq_l=256, pos=True, pos_self=False)
        self.transformerendocer = TransformerEncoder(output_slice=True)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3: unchangeed, new, demolition

        )

    def forward(self, x):
        emb_1, emb_2 = self.embedding(x)
        cls_1, _ = self.transformerendocer(emb_1)
        cls_2, _ = self.transformerendocer(emb_2)
        cls = cls_2 - cls_1
        logits = self.cls_head(cls)  # b 1 3
        return logits


class SiamPTransformer(nn.Module):
    def __init__(self, ):
        super(SiamPTransformer, self).__init__()
        self.embedding = Embedding(embedding_dim=384, seq_l=256, pos=True, pos_self=False)
        self.transformerendocer = TransformerEncoder(output_slice=True)
        self.btsencoder = BTSCrossEncoder(depth=4, mode='org')
        self.cls_head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3: unchangeed, new, demolition
        )

    def forward(self, x):
        emb_1, emb_2 = self.embedding(x)
        cls_1, rep1 = self.transformerendocer(emb_1)
        cls_2, rep2 = self.transformerendocer(emb_2)
        cls = self.btsencoder(rep1, cls_1, rep2, cls_2)
        logits = self.cls_head(cls)  # b 1 3
        return logits


class AsymmetryPCDT(nn.Module):
    def __init__(self, embed_dim=384, seq_len=256):
        super(AsymmetryPCDT, self).__init__()
        self.embedding = PNEmbedding(embed_dim=embed_dim, seq_len=seq_len)
        self.transformerencoder = TransformerEncoder(depth=2, embed_dim=embed_dim, output_slice=False)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 5: {'nochange': 0, 'removed': 1, 'added': 2, 'change': 3, 'color_change': 4}
        )
        self.proj = nn.Linear(seq_len, 1)

    def forward(self, x1, x2):
        x1 = self.embedding(x1)  # 1 seq embed_dim
        x2 = self.embedding(x2)
        x1 = self.transformerencoder(x1)
        x2 = self.transformerencoder(x2)
        logits = self.cls_head(self.proj((x1 - x2).transpose(2, 1)).transpose(2, 1))
        return logits  # 1 1 5


class AsymmetrySiamPT(nn.Module):
    def __init__(self, embed_dim=384, seq_len=256):
        super(AsymmetrySiamPT, self).__init__()
        self.embedding = PNEmbedding(embed_dim=embed_dim, seq_len=seq_len)
        self.transformerencoder = TransformerEncoder(depth=4, embed_dim=embed_dim, output_slice=False)
        self.btsencoder = NaiveCross(depth=4, embed_dim=embed_dim, seq_len=seq_len)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 5: {'nochange': 0, 'removed': 1, 'added': 2, 'change': 3, 'color_change': 4}
        )

    def forward(self, x1, x2):
        x1 = self.embedding(x1)  # 1 seq embed_dim
        x2 = self.embedding(x2)
        x1 = self.transformerencoder(x1)  # 1 seq embed_dim
        x2 = self.transformerencoder(x2)
        x = torch.cat((x1, x2), dim=1)  # 1 2*seq embed_dim
        x = self.btsencoder(x)
        logits = self.cls_head(x)
        return logits  # 1 1 5


if __name__ == "__main__":
    xx = torch.randn(8, 2, 256, 3)
    net = PCDTransformer()
    y = net(xx)
    print(y.shape)
    nett = SiamPTransformer()
    y = nett(xx)
    print(y.shape)
    xxx = torch.randn(1, 100, 3)
    xxxx = torch.randn(1, 200, 3)
    nettt = AsymmetryPCDT()
    y = nettt(xxx, xxxx)
    print(y.shape)
    netttt = AsymmetrySiamPT()
    y = netttt(xxx, xxxx)
    print(y.shape)
