import torch
from torchvision import transforms
from torch.utils.data import random_split
import torch.nn as nn
import time
from setproctitle import setproctitle
import sys
import numpy as np

from src import datasets
from src.utils import set_seed
from src.models import base

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

def main():
    setproctitle("resnet+bert")
    outputpath = "/home/furuya/dl_lecture_competition_pub/logs/trans/"
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=datasets.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=datasets.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn)

    model = base.VQAModel(n_answer=len(train_dataset.dataset.answer2idx)).to(device)

    lr = round(4.704023540427905e-05, 6)
    weight_decay = round(1.2286214982220137e-06, 6)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience, trigger_times = 10, 0

    for epoch in range(100):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"【{epoch + 1}/100】\n"
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
