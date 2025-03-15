import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import tomllib
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 設定ファイルの読み込み
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

result_dir = config["path"]["result_dir"]
model_name = config["path"]["model_name"]
dataset_path = config["path"]["dataset_path"]
gpu = config["gpu"]["gpu_index"]

# デバイスの設定
device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")

# データ変換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データセットの読み込み
test_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 事前学習済みMobileNetV3 Smallモデルをロード
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# 最終層の変更（クラス数に合わせる）
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(test_dataset.classes))
model = model.to(device)

# モデルの重みをロード
model_path = os.path.join(result_dir, model_name)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # strict=Falseでロード
model.eval()

# 予測とラベルの取得
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# クラスごとのデータ数を確認
from collections import Counter
print("Label distribution in test dataset:", Counter(all_labels))

# 混同行列の作成（生データ）
conf_matrix = confusion_matrix(all_labels, all_preds)

# ゼロ除算を防ぐ正規化（行ごとの合計で割る）
row_sums = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_norm = np.divide(conf_matrix, row_sums, where=row_sums != 0)

# 混同行列の可視化と保存
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Normalized)")
plt.savefig(f"{result_dir}/confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Confusion Matrix is saved at {result_dir}/confusion_matrix_normalized.png")
