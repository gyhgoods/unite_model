import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Data set reading
data_dir = "D:\\dataset\\"
batch_size = 32
img_size = (224, 224)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Divide the test set
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)
# 2.Read classification
class_names = train_ds.class_names
num_classes = len(class_names)
print("classify：", class_names)

# 3. data enhancement
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# 4. Add regression label (image brightness)
def add_regression_label(image, label):
    # Regression value = Average brightness of the image (0-1)
    brightness = tf.reduce_mean(image)
    return image, (label, brightness)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), (y, tf.reduce_mean(x)/255.0)))
val_ds = val_ds.map(lambda x, y: (x, (y, tf.reduce_mean(x)/255.0)))
test_ds = test_ds.map(lambda x, y: (x, (y, tf.reduce_mean(x)/255.0)))

# 5.Pre-fetch data to increase training speed
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# 6. CNN combined classification + regression model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)

x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Classification Output
class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name="class_output")(x)
# Regression output
reg_output = tf.keras.layers.Dense(1, activation='linear', name="reg_output")(x)

model = tf.keras.Model(inputs=inputs, outputs=[class_output, reg_output])

model.compile(
    optimizer='adam',
    loss={
        "class_output": "sparse_categorical_crossentropy",
        "reg_output": "mse"
    },
    metrics={
        "class_output": "accuracy",
        "reg_output": "mae"
    }
)

model.summary()

# 7. EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_class_output_accuracy',
    mode='max',
    patience=5,
    restore_best_weights=True
)

# 8. training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stop]
)

# 9. Test set evaluation
results = model.evaluate(test_ds)
class_acc = results[3] * 100  # class_output_accuracy index
reg_mae = results[4]

print(f"Test Classification Accuracy: {class_acc:.2f}%")
print(f"Test Regression MAE: {reg_mae:.4f}")

# 10. confusion matrix + train、val、test graph
y_pred_class = []
y_true_class = []

for images, (labels_class, _) in test_ds:
    preds_class, _ = model.predict(images)
    y_pred_class.extend(np.argmax(preds_class, axis=1))
    y_true_class.extend(labels_class.numpy())

cm = confusion_matrix(y_true_class, y_pred_class)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
test_acc_list = []
for epoch in range(len(history.history['class_output_accuracy'])):
    y_pred_class = []
    y_true_class = []
    for images, (labels_class, _) in test_ds:
        preds_class, _ = model.predict(images, verbose=0)
        y_pred_class.extend(np.argmax(preds_class, axis=1))
        y_true_class.extend(labels_class.numpy())
    test_acc = np.mean(np.array(y_pred_class) == np.array(y_true_class)) * 100
    test_acc_list.append(test_acc)
epochs_range = range(1, len(history.history['class_output_accuracy']) + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs_range, np.array(history.history['class_output_accuracy'])*100, label='Train Accuracy')
plt.plot(epochs_range, np.array(history.history['val_class_output_accuracy'])*100, label='Validation Accuracy')
plt.plot(epochs_range, test_acc_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Classification Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()