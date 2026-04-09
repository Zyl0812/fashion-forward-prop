from math import ceil

import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings, Images
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

from smart_image_similarity.common.paths import CHROMA_DIR, CATALOG_DIR, ensure_runtime_dirs
from smart_image_similarity.common.utils import sorted_alphanum
from smart_image_similarity.similarity.config import CHROMA_INSERT_BATCH_SIZE, ENCODER_MODEL_PATH, IMG_H, IMG_W
from smart_image_similarity.similarity.model import ConvEncoder


class ImageEmbeddingFunction(EmbeddingFunction[Images]):
    def __init__(self, model: ConvEncoder):
        self.model = model

    def __call__(self, inputs: Images) -> Embeddings:
        input_tensor = torch.tensor(np.array(inputs))
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.cpu().numpy()


def get_id2image(main_dir, transform):
    id2image = {}
    img_names = sorted_alphanum([path.name for path in main_dir.iterdir() if path.is_file()])
    with tqdm(total=len(img_names)) as pbar:
        for index, img_name in enumerate(img_names):
            img_path = main_dir / img_name
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            id2image[str(index)] = img_tensor.numpy()
            pbar.update(1)
    return id2image


def get_collection(encoder):
    ensure_runtime_dirs()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="image_similarity",
        embedding_function=ImageEmbeddingFunction(encoder),
    )


def create_embeddings(encoder):
    transform = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])
    print("正在加载图片...")
    id2image = get_id2image(CATALOG_DIR, transform)
    print("图片加载完成！")

    ids = list(id2image.keys())
    images = list(id2image.values())
    collection = get_collection(encoder)

    print("正在写入向量数据库...")
    for batch_idx in range(ceil(len(ids) / CHROMA_INSERT_BATCH_SIZE)):
        start = batch_idx * CHROMA_INSERT_BATCH_SIZE
        end = min((batch_idx + 1) * CHROMA_INSERT_BATCH_SIZE, len(ids))
        collection.upsert(ids=ids[start:end], images=images[start:end])
    print("向量写入完成！")
    return collection


def build_index(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ConvEncoder()
    encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
    encoder.to(device)
    encoder.eval()
    return create_embeddings(encoder)


def search_similar_image_ids(collection, image_tensor, cnt):
    result = collection.query(query_images=[image_tensor.numpy()], n_results=cnt)
    return [int(item) for item in result["ids"][0]]


if __name__ == "__main__":
    build_index()
