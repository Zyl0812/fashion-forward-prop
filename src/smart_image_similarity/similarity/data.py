from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T

from smart_image_similarity.common.utils import sorted_alphanum
from smart_image_similarity.similarity.config import IMG_H, IMG_PATH, IMG_W, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


class ImageDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.image_names = sorted_alphanum([path.name for path in main_dir.iterdir() if path.is_file()])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.main_dir / self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is None:
            raise ValueError("Transform must be provided.")
        img_tensor = self.transform(img)
        return img_tensor, img_tensor


def create_dataset():
    transform = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])
    dataset = ImageDataset(IMG_PATH, transform=transform)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    return train_dataset, val_dataset, test_dataset
