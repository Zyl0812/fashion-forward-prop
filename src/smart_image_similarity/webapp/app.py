import base64
from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from smart_image_similarity.classification.config import CLASSIFIER_MODEL_PATH, classification_names
from smart_image_similarity.classification.model import Classifier
from smart_image_similarity.common.paths import CATALOG_DIR, CHROMA_DIR
from smart_image_similarity.common.settings import SIM_APP_PORT
from smart_image_similarity.denoising.config import DENOISER_MODEL_PATH, NOISE_FACTOR
from smart_image_similarity.denoising.model import Denoiser
from smart_image_similarity.similarity.config import ENCODER_MODEL_PATH
from smart_image_similarity.similarity.embeddings import get_collection, search_similar_image_ids
from smart_image_similarity.similarity.model import ConvEncoder


def _encode_image(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_app():
    app = Flask(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    denoiser = Denoiser()
    denoiser.load_state_dict(torch.load(DENOISER_MODEL_PATH, map_location=device))
    denoiser.to(device)
    denoiser.eval()

    classifier = Classifier()
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    classifier.to(device)
    classifier.eval()

    encoder = ConvEncoder()
    encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
    encoder.to(device)
    encoder.eval()

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise RuntimeError(
            f"Chroma index not found at {CHROMA_DIR}. Run scripts/build_similarity_index.py before starting the app."
        )

    collection = get_collection(encoder)

    @app.route("/")
    def root():
        return jsonify(
            {
                "service": "smart-image-similarity-api",
                "status": "ok",
                "frontend": "removed",
                "endpoints": [
                    "/denoising",
                    "/classification",
                    "/simimages",
                    "/dataset/<filename>",
                ],
            }
        )

    @app.route("/demo")
    def demo():
        return render_template("demo.html")

    @app.route("/dataset/<path:filename>")
    def serve_dataset_image(filename):
        return send_from_directory(CATALOG_DIR, filename)

    @app.post("/denoising")
    def denoise():
        image = request.files["image"]
        image = Image.open(image.stream).convert("RGB")
        image_tensor = transform(image)

        noisy_img = image_tensor + NOISE_FACTOR * torch.randn(*image_tensor.shape)
        noisy_img = torch.clip(noisy_img, 0.0, 1.0).unsqueeze(0)

        with torch.no_grad():
            denoised_image = denoiser(noisy_img.to(device))

        denoised_arr = denoised_image.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
        noisy_arr = noisy_img.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255

        return jsonify(
            {
                "noisy_img": _encode_image(Image.fromarray(noisy_arr.astype("uint8"))),
                "denoised_image": _encode_image(Image.fromarray(denoised_arr.astype("uint8"))),
            }
        )

    @app.post("/classification")
    def classification():
        image = request.files["image"]
        image = Image.open(image.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = classifier(image_tensor.to(device))
        category = classification_names[np.argmax(logits.cpu().detach().numpy())]
        return f"您搜索的商品类型是：{category}"

    @app.post("/simimages")
    def simimages():
        image = request.files["image"]
        image = Image.open(image.stream).convert("RGB")
        image_tensor = transform(image)
        indices_list = search_similar_image_ids(collection, image_tensor, cnt=5)
        return jsonify({"indices_list": indices_list})

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", debug=False, port=SIM_APP_PORT)
