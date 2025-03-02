import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import tomllib  # tomllib を追加（Python 3.11 以上）

# -------------------------
# config.toml を読み込む
with open("config.toml", "rb") as f:  # `rb` モードで開く必要がある
    config = tomllib.load(f)

num_classes = config["hyperparameters"]["num_classes"]
result_dir = config["path"]["result_dir"]
model_name = config["path"]["model_name"]
img_path = config["path"]["img_path"]
gpu = config["gpu"]["gpu_index"]

# -------------------------
# 結果保存ディレクトリの作成（存在しない場合は作成）
os.makedirs(result_dir, exist_ok=True)

# -------------------------
# デバイス設定（GPUがあればCUDAを使用）
device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")

# -------------------------
# MobileNetV3-Small のロード
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes) 
model = model.to(device)
model.eval()

# -------------------------
# 画像の前処理関数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------
# SHAPの準備
def model_forward(x):
    """ PyTorchモデルを SHAP 用にラップする関数 """
    model.eval()
    with torch.no_grad():
        return model(x.to(device)).cpu().numpy()

# -------------------------
# 画像の読み込みと処理
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)  # バッチ次元を追加

# -------------------------
# SHAP の `DeepExplainer` を使用
background = torch.cat([input_tensor] * 10, dim=0)  # SHAP用のバックグラウンドデータ（適宜変更）
explainer = shap.DeepExplainer(model_forward, background)
shap_values = explainer.shap_values(input_tensor)

# -------------------------
# SHAPの可視化
plt.figure(figsize=(10, 5))

# 元画像の表示
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")

# SHAP値を可視化
shap_img = np.transpose(shap_values[0], (1, 2, 0))  # (C, H, W) → (H, W, C)
shap_img = np.mean(shap_img, axis=2)  # RGB の平均を取る

plt.subplot(1, 2, 2)
plt.imshow(shap_img, cmap="jet")
plt.axis("off")
plt.title("SHAP Explanation")

# -------------------------
# 画像を保存
shap_img_path = os.path.join(result_dir, "shap_visualization.png")
plt.savefig(shap_img_path, dpi=300, bbox_inches="tight")
print(f"SHAP 可視化画像を {shap_img_path} に保存しました。")

# -------------------------
# 画像を表示
plt.show()
