import torch
import torch.nn as nn
from module_pw_pg import Embedding, EmbeddingPosself, TransformerEncoder


class PCDTransformer(nn.Module):
    def __init__(self, ):
        super(PCDTransformer, self).__init__()
        self.embedding = Embedding(embedding_dim=512, group=16, pos=True)
        self.transformerendocer = TransformerEncoder(output_slice=True)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
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


class PCDTransformerPosself(nn.Module):
    def __init__(self, ):
        super(PCDTransformerPosself, self).__init__()
        self.embedding = EmbeddingPosself(embedding_dim=512, pos=True)
        self.transformerendocer = TransformerEncoder(output_slice=True)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3: unchangeed, new, demolition

        )

    def forward(self, x, pos):
        emb_1, emb_2 = self.embedding(x, pos)
        cls_1, _ = self.transformerendocer(emb_1)
        cls_2, _ = self.transformerendocer(emb_2)
        cls = cls_2 - cls_1
        logits = self.cls_head(cls)  # b 1 3
        return logits


if __name__ == "__main__":
    xx = torch.randn(8, 2, 16, 16, 3)
    poss = torch.randn(8, 2, 16, 3)
    net = PCDTransformer()
    y = net(xx)
    print(y.shape)
    nett = PCDTransformerPosself()
    y = nett(xx, poss)
    print(y.shape)
