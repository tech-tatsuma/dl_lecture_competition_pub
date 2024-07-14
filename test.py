
from setproctitle import setproctitle
from torchvision import transforms
import torch
import numpy as np

from src import datasets
from src.utils import set_seed
from src.models import base

def test(model_path: str):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = datasets.VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn)

    model = base.VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    np.save("submission.npy", submission)




if __name__ == "__main__":
    setproctitle("test")
    model_path = "best_model.pth"
    test(model_path=model_path)
