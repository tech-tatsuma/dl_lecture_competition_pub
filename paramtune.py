import re
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from setproctitle import setproctitle
from torch.utils.data import random_split

from src import datasets
from src.models import base

from src.utils import set_seed

import sys
import requests
import json

import optuna

# 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

# 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()

    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

# モデルの評価を行う関数の実装
def eval(model, dataloader, optimizer, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()

    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for image, question, answers, mode_answer in dataloader:
            image, question, answers, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())
            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)
    return total_loss / len(dataloader), total_acc / len(dataloader)

# Optunaの目的関数
def objective(trial):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=datasets.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=datasets.collate_fn)

    model = base.VQAModel(n_answer=len(train_dataset.dataset.answer2idx)).to(device)

    lr = round(trial.suggest_loguniform('lr', 1e-5, 1e-3), 6)
    weight_decay = round(trial.suggest_loguniform('weight_decay', 1e-6, 1e-4), 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience, trigger_times = 5, 0

    for epoch in range(20):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"【{epoch + 1}/10】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"Val Loss {val_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}\n"
              f"Val Acc {val_acc:.4f}")
        sys.stdout.flush()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    return best_val_acc

def main():
    setproctitle("resnet+bert")
    outputpath = "/home/furuya/dl_lecture_competition_pub/logs/trans/"

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # ベストパラメータで最終学習
    best_params = trial.params
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    test_dataset = datasets.VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset.dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=datasets.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=datasets.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn)

    model = base.VQAModel(n_answer=len(train_dataset.dataset.answer2idx)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience, trigger_times = 10, 0

    for epoch in range(20):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"【{epoch + 1}/10】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"Val Loss {val_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}\n"
              f"Val Acc {val_acc:.4f}")
        sys.stdout.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping after {epoch + 1} epochs!")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), outputpath + "model.pth")
    np.save(outputpath + "submission.npy", submission)

if __name__ == "__main__":
    main()
