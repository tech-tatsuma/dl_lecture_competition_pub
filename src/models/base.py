import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel
import timm

from ..utils import set_seed

set_seed(42)
    
class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        # ImageNetで事前学習されたResNet18を使用
        # self.resnet = models.resnet18(pretrained=True)
        # self.resnet = models.resnet50(pretrained=True)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=768)
        self.vit.head = nn.Identity()
        # 事前学習モデルの最後の層を取り除く
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # BERTモデルのロード
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        with torch.no_grad():
            # ダミー入力を使って出力サイズを計算
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.vit(dummy_input)
            resnet_output_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        combined_input_size = resnet_output_size + self.bert.config.hidden_size

        # 画像とテキストの特徴量を結合して処理する全結合層
        # ResNetからの出力サイズとBERTからの出力サイズを足し合わせて入力サイズとする
        self.fc = nn.Sequential(
            nn.Linear(combined_input_size, 512),  # ResNetの出力512, BERTの出力768 (通常)
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

        # Xavier初期化を適用
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image, question_ids):
        # 画像の特徴量を抽出
        image_feature = self.vit(image)

        # 質問の特徴量を抽出
        question_output = self.bert(input_ids=question_ids)
        question_feature = question_output.pooler_output  # BERTからの最終的な特徴量を使用

        # 画像特徴量と質問特徴量を結合
        combined_features = torch.cat([image_feature, question_feature], dim=1)

        # 最終出力を計算
        output = self.fc(combined_features)

        return output