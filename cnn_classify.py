import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score

# dataset
data_dir = "D:\\dataset\\"
img_size = (224, 224)
batch_size = 32
epochs = 50
# =========================
# 1.  train/val/test
# =========================
full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_ds = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_ds = remaining.take(val_size)
test_ds = remaining.skip(val_size)

class_names = full_dataset.class_names
num_classes = len(class_names)
print("classification：", class_names)

# 2. Data augmentation and preprocessing
train_ds = train_ds.map(lambda x, y: (tf.image.random_flip_left_right(x),
                                      y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# 3. Create Sequential CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 4. EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=10,
    restore_best_weights=True
)

# 5. model training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stop]
)

# 7. Confusion Matrix
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Set')
plt.show()

# 8. Epoch
train_acc = np.array(history.history['accuracy']) * 100
val_acc = np.array(history.history['val_accuracy']) * 100
epochs = len(train_acc)
test_acc_list = []
print(epochs)
for epoch in range(epochs):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    test_acc = accuracy_score(y_true, y_pred) * 100
    test_acc_list.append(test_acc)

epochs_range = range(1, epochs+1)

plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.plot(epochs_range, test_acc_list, label='Test Accuracy', linestyle='--', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Classification Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()