import re
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src import datasets
from src.models import base

from src.utils import set_seed

# 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    # accuracyを格納する変数の初期化
    total_acc = 0.

    # バッチの中のそれぞれのデータに対してループを回す
    for pred, answers in zip(batch_pred, batch_answers): # バッチ内の予測と正解データのペアに対してループ
        # 個々のデータの正解率を格納する変数を0で初期化
        acc = 0.

        # 各予測値に対して
        for i in range(len(answers)):
            # 現在の予測値と一致する回答の数をカウントする変数を0で初期化
            num_match = 0
            # 同じ回答郡内で他の回答の数をカウントする変数を0で初期化
            for j in range(len(answers)):
                if i == j:
                    continue # 同じインデックスの回答は比較しない
                # 予測値が回答と一致するかチェック
                if pred == answers[j]:
                    # 一致する場合，カウントを1増やす
                    num_match += 1
            # 一致数を3で割ったものと1の小さい方を正解率に加算
            acc += min(num_match / 3, 1)

        # 各データの平均正解率(10人の回答者)を全体の正解率に加算
        total_acc += acc / 10

    # バッチ全体の平均正解率を計算して返す
    return total_acc / len(batch_pred)


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):

    # モデルを学習モードに設定
    model.train()

    # 損失の値を格納する変数を初期化
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    # 学習の開始時間を計測
    start = time.time()

    # バッチループ
    for image, question, answers, mode_answer in dataloader:

        # 全データをデバイスに転送
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        # モデルにデータを入力して予測値を取得
        pred = model(image, question)
        # 損失関数の計算
        loss = criterion(pred, mode_answer.squeeze())

        # 勾配の初期化と逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 損失の値を加算
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

# モデルの評価を行う関数の実装
def eval(model, dataloader, optimizer, criterion, device):

    # モデルを評価モードに設定
    model.eval()

    # 損失の値を格納する変数を初期化
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    # 評価の開始時間を計測
    start = time.time()

    # バッチループ
    for image, question, answers, mode_answer in dataloader:
        # データをデバイスに転送
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        # モデルにデータを入力して予測値を取得
        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        # 損失の値を加算
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定とシードの固定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 画像データの前処理の定義
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # データセットの作成
    train_dataset = datasets.VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = datasets.VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    # データローダーの作成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデルの作成
    model = base.VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

    # ハイパーパラメータの設定
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # エポックのループ
    for epoch in range(num_epoch):

        # モデルの学習
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        # 学習結果の出力
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
