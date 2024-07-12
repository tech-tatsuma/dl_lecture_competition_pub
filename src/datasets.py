import re
from statistics import mode

import torch
import pandas
from PIL import Image
import numpy as np
import sys

from torchtext.transforms import CLIPTokenizer
from torch.nn.utils.rnn import pad_sequence

from .utils import set_seed

set_seed(42)

# テキストの前処理を行う関数
def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer # answerを使用するかどうかのフラグ

        self.tokenizer = CLIPTokenizer(merges_path="http://download.pytorch.org/models/text/clip_merges.bpe", encoder_json_path="http://download.pytorch.org/models/text/clip_encoder.json")

        # # question / answerの辞書を作成
        # self.question2idx = {}
        # self.answer2idx = {}
        # self.idx2question = {}
        # self.idx2answer = {}

        # # 質問文に含まれる単語を辞書に追加
        # for question in self.df["question"]:
        #     question = process_text(question) # 質問文の前処理
        #     words = question.split(" ") # 質問文を単語に分割
        #     for word in words:
        #         if word not in self.question2idx:
        #             self.question2idx[word] = len(self.question2idx) # 辞書に単語を追加
        # self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        # if self.answer:
        #     # 回答に含まれる単語を辞書に追加
        #     for answers in self.df["answers"]:
        #         for answer in answers:
        #             word = answer["answer"]
        #             word = process_text(word) # 回答の前処理
        #             if word not in self.answer2idx:
        #                 self.answer2idx[word] = len(self.answer2idx) # 辞書に単語を追加
        #     self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            self.answer2idx = {}
            self.idx2answer = {}

            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx) # 辞書に単語を追加
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        # self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        # self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        # 画像ファイルを開く
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        # 画像の前処理を行う
        image = self.transform(image)
        # # one-hot表現のための配列（未知語用の要素を追加）
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        # # 質問文を単語に分割
        # # question_words = self.df["question"][idx].split(" ")
        # question_words = process_text(self.df["question"][idx]).split(" ")
        # for word in question_words:
        #     try:
        #         question[self.question2idx[word]] = 1  # one-hot表現に変換
        #     except KeyError:
        #         question[-1] = 1  # 未知語

        # 質問文のトークナイズ
        question = process_text(self.df["question"][idx])
        question_tokens = self.tokenizer(question)
        question_ids = torch.tensor([int(token) for token in question_tokens], dtype=torch.long)

        if self.answer:
            # 回答のリスト
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            # 最頻値の取得
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            # 回答ありの場合の返り値
            # return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)
            return image, question_ids, torch.tensor(answers, dtype=torch.long), torch.tensor([mode_answer_idx], dtype=torch.long)

        else:
            # 回答なしの場合の返り値
            # return image, torch.Tensor(question)
            return image, question_ids

    def __len__(self):
        """
        データセットの長さを返す関数
        """
        return len(self.df)
    
def collate_fn(batch):
    images, questions, answers, mode_answers = zip(*batch)
    images = torch.stack(images, dim=0)
    questions = pad_sequence(questions, batch_first=True, padding_value=0)
    answers = torch.stack(answers, dim=0)
    mode_answers = torch.stack(mode_answers, dim=0)
    return images, questions, answers, mode_answers