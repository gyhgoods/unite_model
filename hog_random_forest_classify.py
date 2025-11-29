import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import random
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import seaborn as sns
# ============================
# 数据增强函数（适合 HOG）
# ============================
def augment_image(img):
    # 1. 轻微旋转（±3°）
    angle = random.uniform(-3, 3)
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # 转为 int16 防止溢出
    img = img.astype(np.int16)

    # 2. 亮度变化
    delta = random.randint(-20, 20)
    img = img + delta

    # 3. 加一点噪声
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = img + noise

    # 限制范围并转回 uint8
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

# ============================
# HOG 特征提取
# ============================
def extract_hog_features(img):
    features, hog_img = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def extract_lbp(img):
    radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(img, n_points, radius, method="uniform")

    # LBP 59维直方图特征（固定尺寸）
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=59,
        range=(0, 59),
        density=True
    )
    return hist

def extract_features(img):
    hog_f = extract_hog_features(img)   # HOG 特征
    lbp_f = extract_lbp(img)            # LBP 特征
    features = np.hstack([hog_f, lbp_f])  # 拼接成最终特征
    return features

def load_dataset(dataset_path):
    X, y = [], []

    classes = os.listdir(dataset_path)

    for label in classes:
        folder = os.path.join(dataset_path, label)

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))

            # ----------------------------------------
            # 1. 添加原图特征
            # ----------------------------------------
            X.append(extract_features(img))
            y.append(label)

            # ----------------------------------------
            # 2. 数据增强次数
            # ----------------------------------------
            if label == "metal":
                aug_times = 3      # 👉 金属类 3 倍增强
            else:
                aug_times = 1      # 其他类只增强 1 倍

            # ----------------------------------------
            # 3. 执行数据增强
            # ----------------------------------------
            for _ in range(aug_times):
                aug = augment_image(img)
                X.append(extract_features(aug))
                y.append(label)

    return np.array(X), np.array(y)

# ============================
# 主流程
# ============================
dataset_path = "D:\\dataset\\"
X, y = load_dataset(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
train = model.predict(X_train)
ac2 = accuracy_score(y_train, train)
print('train Accuracy:',ac2)
print("Test Accuracy:", acc)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

train = model.predict(X_train)
ac2 = accuracy_score(y_train, train)
print('train Accuracy:',ac2)
print(len(X_train))
# 获取类别名顺序
labels = ['battery','biological','cardboard','metal']
cm = confusion_matrix(y_test, pred, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()