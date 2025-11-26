import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. parameter setting
DATASET_DIR = "D:\\dataset\\"

IMAGE_SIZE = (128, 128)
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9
RANDOM_STATE = 42

# 2.feature extraction
X = []
y = []
class_names = []

for folder_name in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    class_label = folder_name
    class_names.append(class_label)

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"error picture：{img_path}")
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        feat = hog(
            img,
            pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK,
            orientations=ORIENTATIONS
        )
        X.append(feat)
        y.append(class_label)

X = np.array(X)
y = np.array(y)
print(f"data loading， {len(X)} picture，latitude = {X.shape[1]}")
print("type：", class_names)

# 3.divide dataset
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
)
# 60% train + 20% val + 20% test

print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# 4. RandomForest training
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_STATE
)

clf.fit(X_train, y_train)

# 5. calculate accuracy
acc_train = accuracy_score(y_train, clf.predict(X_train))
acc_val   = accuracy_score(y_val, clf.predict(X_val))
acc_test  = accuracy_score(y_test, clf.predict(X_test))

print("train Accuracy: {:.2f}%".format(acc_train*100))
print("val Accuracy: {:.2f}%".format(acc_val*100))
print("test Accuracy: {:.2f}%".format(acc_test*100))

# 6. confusion matrix
y_pred_test = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test, labels=class_names)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("forecast")
plt.ylabel("reality")
plt.title("test set confusion matrix")
plt.show()