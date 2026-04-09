from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T

from smart_image_similarity.classification.config import (
    FASHION_LABELS_PATH,
    IMG_H,
    IMG_PATH,
    IMG_W,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from smart_image_similarity.common.utils import sorted_alphanum


class ImageLabelDataset(Dataset):
    def __init__(self, main_dir, label_path, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.image_names = sorted_alphanum([path.name for path in main_dir.iterdir() if path.is_file()])
        labels = pd.read_csv(label_path)
        self.labels_dict = dict(zip(labels["id"], labels["target"]))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.main_dir / self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is None:
            raise ValueError("Transform must be provided.")
        img_tensor = self.transform(img)
        target = self.labels_dict[idx]
        return img_tensor, target


def create_dataset():
    transform = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])
    dataset = ImageLabelDataset(main_dir=IMG_PATH, label_path=FASHION_LABELS_PATH, transform=transform)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    return train_dataset, val_dataset, test_dataset
