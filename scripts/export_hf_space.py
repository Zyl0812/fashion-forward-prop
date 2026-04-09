from pathlib import Path
import shutil

from smart_image_similarity.common.settings import PROJECT_ROOT


HF_README = """---
title: Smart Image Similarity
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 5000
---

# 智图寻宝

这是 Hugging Face Space 的精简部署目录，只包含推理服务运行所需的源码、模型、图片数据和预构建 Chroma 索引。
"""


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    project_root = Path(PROJECT_ROOT)
    target = project_root / "deploy" / "huggingface"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    required_paths = [
        project_root / "src",
        project_root / "assets" / "models",
        project_root / "assets" / "data",
        project_root / "artifacts" / "chroma",
        project_root / "requirements.txt",
        project_root / "deploy" / "hf.Dockerfile",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required export input: {path}")

    copy_path(project_root / "src", target / "src")
    copy_path(project_root / "assets" / "models", target / "assets" / "models")
    copy_path(project_root / "assets" / "data", target / "assets" / "data")
    copy_path(project_root / "artifacts" / "chroma", target / "artifacts" / "chroma")
    copy_path(project_root / "requirements.txt", target / "requirements.txt")
    copy_path(project_root / "deploy" / "hf.Dockerfile", target / "Dockerfile")

    (target / "app.py").write_text(
        "from smart_image_similarity.webapp.app import create_app\n\n"
        "app = create_app()\n\n"
        "if __name__ == '__main__':\n"
        "    app.run(host='0.0.0.0', port=5000, debug=False)\n",
        encoding="utf-8",
    )
    (target / "README.md").write_text(HF_README, encoding="utf-8")
    print(f"Exported Hugging Face Space to {target}")


if __name__ == "__main__":
    main()
