# Fashion Forward Prop

**一个可交互的 CNN 前向传播教学演示** — 选择一张时尚商品图片，观察卷积神经网络如何逐层提取特征，依次完成去噪、分类与相似检索。

🚀 **在线体验**：[huggingface.co/spaces/zyl0812/fashion-forward-prop](https://huggingface.co/spaces/zyl0812/fashion-forward-prop)

---

## 目录

- [项目概览](#项目概览)
- [快速开始](#快速开始)
- [使用自己的数据集](#使用自己的数据集)
- [三个模块的原理与训练](#三个模块的原理与训练)
  - [模块一：图像去噪（自编码器）](#模块一图像去噪自编码器)
  - [模块二：商品分类（CNN 分类器）](#模块二商品分类cnn-分类器)
  - [模块三：相似检索（ConvEncoder + 向量索引）](#模块三相似检索convencoder--向量索引)
- [构建检索索引](#构建检索索引)
- [启动前端演示](#启动前端演示)
- [后端 API](#后端-api)
- [项目结构](#项目结构)

---

## 项目概览

本项目用真实训练好的模型 + Canvas 2D 动画，直观展示 CNN 前向传播过程：

```
输入图片
  │
  ├─ 分类器路径（Coral 粒子流动）
  │    Conv1 → Pool1 → Conv2 → Pool2 → FC → 输出类别
  │
  └─ 编码器路径（Teal 粒子流动）
       Conv → Conv → Conv → Conv → Conv → 512维特征向量 → Top-5 相似商品
```

三个推理模块全部使用真实模型，不是 Mock 数据：

| 模块 | 模型结构 | 输出 |
|------|----------|------|
| 去噪器 | 卷积自编码器（编码器-解码器） | 加噪图 vs 去噪图 |
| 分类器 | 轻量 CNN（Conv×2 + FC） | 5类商品标签 |
| 编码器 | ConvEncoder（5层卷积） | 512维嵌入 → Top-5 相似图 |

---

## 快速开始

**环境要求**：Python 3.10+

```bash
git clone https://github.com/Zyl0812/fashion-forward-prop.git
cd fashion-forward-prop
pip install -r requirements.txt
```

如果已有训练好的权重和 Chroma 索引，直接启动：

```bash
PYTHONPATH=src python src/smart_image_similarity/webapp/app.py
```

打开 http://localhost:5000 即可体验演示。

> **没有权重？** 按下方步骤用自己的数据集训练，或从 HuggingFace Space 下载预训练权重。

---

## 使用自己的数据集

### 1. 准备图片

将你的图片放入 `assets/data/catalog/`，命名为连续整数（`0.jpg`、`1.jpg`、…）：

```
assets/data/catalog/
├── 0.jpg
├── 1.jpg
├── 2.jpg
└── ...
```

图片格式：JPEG，任意分辨率（训练时自动缩放为 64×64）。

### 2. 准备标签文件

在 `assets/data/` 创建 `fashion-labels.csv`：

```csv
image_id,label
0,上身衣服
1,鞋
2,包
3,下身衣服
4,手表
```

支持的类别可在 `src/smart_image_similarity/classification/config.py` 自定义：

```python
classification_names = ['上身衣服', '鞋', '包', '下身衣服', '手表']
```

### 3. 验证数据加载

```bash
PYTHONPATH=src python -c "
from smart_image_similarity.common.dataset import FashionDataset
ds = FashionDataset()
print(f'数据集大小: {len(ds)} 张图片')
print(f'第一张: {ds[0][0].shape}, 标签: {ds[0][1]}')
"
```

---

## 三个模块的原理与训练

### 模块一：图像去噪（自编码器）

**原理**

去噪自编码器（Denoising Autoencoder）是一种无监督学习方法。训练时主动向输入图片添加高斯噪声，让网络学习从噪声图恢复干净图的能力：

```
原始图 → 加噪 → [编码器] → 瓶颈特征 → [解码器] → 去噪图
                                              ↑
                              与原始图计算 MSE Loss
```

本项目的去噪器结构（`src/smart_image_similarity/denoising/model.py`）：

```
输入 (3,64,64)
  → Conv(3→16) + ReLU + MaxPool    # 编码：(16,32,32)
  → Conv(16→4) + ReLU + MaxPool    # 编码：(4,16,16)
  → ConvTranspose(4→16) + ReLU     # 解码：(16,32,32)
  → ConvTranspose(16→3) + Sigmoid  # 解码：(3,64,64)
```

**训练**

```bash
PYTHONPATH=src python scripts/train_denoiser.py
```

训练参数（`src/smart_image_similarity/denoising/config.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NOISE_FACTOR` | 0.3 | 高斯噪声强度 |
| `EPOCHS` | 20 | 训练轮数 |
| `BATCH_SIZE` | 32 | 批大小 |
| `LR` | 1e-3 | 学习率 |

权重保存至 `assets/models/denoiser.pt`。

---

### 模块二：商品分类（CNN 分类器）

**原理**

轻量 CNN 分类器，对 64×64 的商品图片做 5 分类：

```
输入 (3,64,64)
  → Conv(3→8, 3×3) + ReLU + MaxPool(2)    # (8,32,32)
  → Conv(8→16, 3×3) + ReLU + MaxPool(2)   # (16,16,16)
  → Flatten                                # 4096
  → FC(4096→128) + ReLU
  → FC(128→5)                              # 5类 logits
```

损失函数：CrossEntropyLoss，优化器：Adam。

**训练**

```bash
PYTHONPATH=src python scripts/train_classifier.py
```

训练参数（`src/smart_image_similarity/classification/config.py`）：

| 参数 | 默认值 |
|------|--------|
| `EPOCHS` | 30 |
| `BATCH_SIZE` | 64 |
| `LR` | 1e-3 |

权重保存至 `assets/models/classifier.pt`。

---

### 模块三：相似检索（ConvEncoder + 向量索引）

**原理**

相似检索分两步：

**第一步：训练 ConvEncoder**

ConvEncoder 将图片压缩为 512 维嵌入向量，同类商品的向量在空间中靠近，不同类商品的向量远离：

```
输入 (3,64,64)
  → Conv(3→16)   + BN + ReLU + MaxPool    # (16,32,32)
  → Conv(16→32)  + BN + ReLU + MaxPool    # (32,16,16)
  → Conv(32→64)  + BN + ReLU + MaxPool    # (64,8,8)
  → Conv(64→128) + BN + ReLU + MaxPool    # (128,4,4)
  → Conv(128→256)+ BN + ReLU + MaxPool    # (256,2,2)
  → Flatten → FC(1024→512)                # 512维向量
```

**第二步：构建 Chroma 向量索引**

用训练好的 ConvEncoder 对所有图片提取嵌入，写入 [Chroma](https://www.trychroma.com/) 向量数据库。查询时计算查询图的嵌入，在数据库中找余弦最近邻：

```
查询图 → ConvEncoder → 512维向量 → Chroma 近邻搜索 → Top-5 图片 ID
```

**训练**

```bash
PYTHONPATH=src python scripts/train_similarity.py
```

权重保存至 `assets/models/deep_encoder.pt`。

---

## 构建检索索引

三个模型训练完成后，将全部图片嵌入写入 Chroma：

```bash
PYTHONPATH=src python scripts/build_similarity_index.py
```

索引保存至 `artifacts/chroma/`。**这一步是启动后端的前提**，若索引不存在服务无法启动。

---

## 启动前端演示

```bash
PYTHONPATH=src python src/smart_image_similarity/webapp/app.py
```

访问 http://localhost:5000，演示页面流程：

1. **选图** — 点击任意缩略图
2. **转场** — 图片飞入 Canvas 输入区
3. **Phase 1（Coral）** — 粒子流过 Conv1 → Pool1，调用 `/denoising`，展示去噪结果
4. **Phase 2（Terracotta）** — 粒子流过 Conv2 → Pool2 → FC → Output，调用 `/classification`，展示分类结果
5. **Phase 3（Teal）** — 粒子流过编码器各层生成 512 维向量，调用 `/simimages`，展示 Top-5 相似商品
6. **重置** — 点击「← 重新选择」回到首页

前端为单文件实现（`src/smart_image_similarity/webapp/templates/demo.html`），无任何前端框架依赖，可直接阅读和修改。

---

## 后端 API

| 方法 | 路径 | 请求 | 响应 |
|------|------|------|------|
| `GET` | `/` | — | CNN 教学演示页面 |
| `GET` | `/dataset/<filename>` | — | 商品图片文件 |
| `POST` | `/denoising` | `form-data: image=<file>` | `{"noisy_img": "<base64>", "denoised_image": "<base64>"}` |
| `POST` | `/classification` | `form-data: image=<file>` | `"您搜索的商品类型是：上身衣服"` |
| `POST` | `/simimages` | `form-data: image=<file>` | `{"indices_list": [42, 118, 7, ...]}` |

---

## 项目结构

```
├── src/smart_image_similarity/
│   ├── classification/     # 分类器模型、训练引擎、数据加载
│   ├── denoising/          # 去噪器模型、训练引擎
│   ├── similarity/         # ConvEncoder、Chroma 检索、训练引擎
│   ├── common/             # 路径配置、数据集、工具函数
│   └── webapp/
│       ├── app.py          # Flask 入口（API + 路由）
│       └── templates/
│           └── demo.html   # CNN 教学演示前端（单文件）
├── scripts/
│   ├── train_denoiser.py
│   ├── train_classifier.py
│   ├── train_similarity.py
│   └── build_similarity_index.py
├── assets/
│   ├── data/catalog/       # 商品图片（0.jpg ~ N.jpg）
│   └── models/             # 训练好的模型权重（不含于本仓库）
├── artifacts/chroma/       # Chroma 向量索引（build 后生成，不含于本仓库）
└── requirements.txt
```

---

## License

MIT
