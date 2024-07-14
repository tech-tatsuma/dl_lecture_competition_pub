import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel
import sys
import math

from ..utils import set_seed

set_seed(42)

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        d_tensor = k.size(-1)  # テンソルの次元数を取得
        score = (q @ k.transpose(-2, -1)) / math.sqrt(d_tensor)  # scaled dot product

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)
        self.fc = nn.Linear(n_head * self.d_v, d_model)

        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size = q.size(0)

        q = self.w_qs(q).view(batch_size, n_head, d_k).transpose(0, 1)  # [n_head, batch_size, d_k]
        k = self.w_ks(k).view(batch_size, n_head, d_k).transpose(0, 1)  # [n_head, batch_size, d_k]
        v = self.w_vs(v).view(batch_size, n_head, d_v).transpose(0, 1)  # [n_head, batch_size, d_v]

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(0, 1).contiguous().view(batch_size, -1)
        output = self.fc(output)
        return output
    
class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        # ImageNetで事前学習されたResNet18を使用
        self.resnet = models.resnet18(pretrained=True)
        # 事前学習モデルの最後の層を取り除く
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # BERTモデルのロード
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.multihead_attn = MultiHeadAttention(d_model=512 + self.bert.config.hidden_size, n_head=8)

        # 画像とテキストの特徴量を結合して処理する全結合層
        # ResNetからの出力サイズとBERTからの出力サイズを足し合わせて入力サイズとする
        self.fc = nn.Sequential(
            nn.Linear(512 + self.bert.config.hidden_size, 512),  # ResNetの出力512, BERTの出力768 (通常)
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question_ids):
        # 画像の特徴量を抽出
        image_feature = self.resnet(image)
        image_feature = image_feature.view(image_feature.size(0), -1)  # バッチサイズに合わせてフラット化

        # 質問の特徴量を抽出
        question_output = self.bert(input_ids=question_ids)
        question_feature = question_output.pooler_output  # BERTからの最終的な特徴量を使用

        # 画像特徴量と質問特徴量を結合
        combined_features = torch.cat([image_feature, question_feature], dim=1)
        attn_output = self.multihead_attn(combined_features, combined_features, combined_features)
        attn_output = attn_output.squeeze(1)

        # 最終出力を計算
        # output = self.fc(combined_features)
        output = self.fc(attn_output)

        return output