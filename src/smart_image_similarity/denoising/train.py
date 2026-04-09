import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from smart_image_similarity.common.paths import ensure_runtime_dirs
from smart_image_similarity.common.utils import seed_everything
from smart_image_similarity.denoising.config import (
    DENOISER_MODEL_PATH,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
)
from smart_image_similarity.denoising.data import create_dataset
from smart_image_similarity.denoising.engine import train_epoch, val_step
from smart_image_similarity.denoising.model import Denoiser


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED)
    ensure_runtime_dirs()

    train_dataset, val_dataset, _ = create_dataset()
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=True)

    model = Denoiser().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_val_loss = float("inf")
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(model, device, train_loader, loss_fn, optimizer)
        val_loss = val_step(model, device, val_loader, loss_fn)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), DENOISER_MODEL_PATH)
        else:
            print("验证损失未下降，跳过保存!")
    print("训练完成!")


if __name__ == "__main__":
    main()
