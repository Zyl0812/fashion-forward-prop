import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from smart_image_similarity.classification.config import (
    CLASSIFIER_MODEL_PATH,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    TEST_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
)
from smart_image_similarity.classification.data import create_dataset
from smart_image_similarity.classification.engine import train_epoch, val_step
from smart_image_similarity.classification.model import Classifier
from smart_image_similarity.common.paths import ensure_runtime_dirs
from smart_image_similarity.common.utils import seed_everything


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED)
    ensure_runtime_dirs()

    train_dataset, _, test_dataset = create_dataset()
    train_loader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, TEST_BATCH_SIZE, shuffle=False)

    classifier = Classifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)

    min_val_loss = float("inf")
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(classifier, device, train_loader, loss_fn, optimizer)
        val_loss = val_step(classifier, device, test_loader, loss_fn)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(classifier.state_dict(), CLASSIFIER_MODEL_PATH)
            print("验证集损失下降，保存模型！")
        else:
            print("验证集损失未下降，不保存模型！")
    print("训练完成!")


if __name__ == "__main__":
    main()
