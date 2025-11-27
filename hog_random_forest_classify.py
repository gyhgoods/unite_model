import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# -----------------------------
# dataset
# -----------------------------
DATASET_DIR = "D:\\dataset\\"
IMAGE_SIZE = (224,224)
PIXELS_PER_CELL = (16,16)
CELLS_PER_BLOCK = (2,2)
ORIENTATIONS = 9

ROTATE_MAX = 10
TRANSLATE_MAX = 5
SCALE_MIN, SCALE_MAX = 0.9, 1.1

PCA_COMPONENTS = 220
RANDOM_STATE = 42

# 1.Image enhancement function
def augment_image(img):
    h, w = img.shape
    angle = random.uniform(-ROTATE_MAX, ROTATE_MAX)
    M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M_rot, (w,h))
    tx = random.uniform(-TRANSLATE_MAX, TRANSLATE_MAX)
    ty = random.uniform(-TRANSLATE_MAX, TRANSLATE_MAX)
    M_trans = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img, M_trans, (w,h))
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img = cv2.resize(img, (w,h))
    return img

# 2.HOG Feature Extraction
X = []
y = []
class_names = []

for folder_name in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    class_names.append(folder_name)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error image:", img_path)
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img_aug = augment_image(img)
        feat = hog(img_aug, pixels_per_cell=PIXELS_PER_CELL,
                   cells_per_block=CELLS_PER_BLOCK,
                   orientations=ORIENTATIONS,
                   block_norm='L2-Hys')
        X.append(feat)
        y.append(folder_name)

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} samples, HOG dim = {X.shape[1]}")

# 3.train/val/test = 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 4.standardization + PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=PCA_COMPONENTS, whiten=True, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 5.Random Forest Training
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=3,
    random_state=RANDOM_STATE
)

clf.fit(X_train_pca, y_train)

# 6.Accuracy
train_acc = accuracy_score(y_train, clf.predict(X_train_pca))
val_acc = accuracy_score(y_val, clf.predict(X_val_pca))
test_acc = accuracy_score(y_test, clf.predict(X_test_pca))

print("\n===== Accuracy =====")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Val Accuracy:   {val_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")

# 6.Test Confusion Matrix
y_test_pred = clf.predict(X_test_pca)
cm = confusion_matrix(y_test, y_test_pred, labels=class_names)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
