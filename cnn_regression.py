import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, confusion_matrix
import seaborn as sns

# set dataset
data_dir = "D:\\dataset\\"
img_size = (224, 224)
batch_size = 32
epochs = 50

# 1. dataset loading
full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
# train/val/test
dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_ds = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_ds = remaining.take(val_size)
test_ds = remaining.skip(val_size)

# 2. Construct regression labels (average brightness)
def add_brightness_label(image, label):
    # normalization
    image = tf.cast(image, tf.float32) / 255.0
    brightness = tf.reduce_mean(image)  #mean brightness
    return image, brightness

train_ds = train_ds.map(lambda x, y: add_brightness_label(x, y))
val_ds = val_ds.map(lambda x, y: add_brightness_label(x, y))
test_ds = test_ds.map(lambda x, y: add_brightness_label(x, y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# 3. Build a CNN regression model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# 4. EarlyStopping and MAE Callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=10,
    restore_best_weights=True
)
class TestMAECallback(tf.keras.callbacks.Callback):
    def __init__(self, test_ds):
        super().__init__()
        self.test_ds = test_ds
        self.test_mae = []

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []
        for images, brightness in self.test_ds:
            preds = self.model.predict(images, verbose=0)
            y_pred.extend(preds.flatten())
            y_true.extend(tf.reduce_mean(images, axis=[1,2,3]).numpy().flatten())
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        self.test_mae.append(mae)
test_callback = TestMAECallback(test_ds)

# 5. training model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stop,test_callback]
)

# 6. test predict
y_true = []
y_pred = []

for images, brightness in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(preds.flatten())
    brightness_batch = tf.reduce_mean(images, axis=[1,2,3])
    y_true.extend(brightness_batch.numpy().flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
test_mae = mean_absolute_error(y_true, y_pred)
print(f"Test MAE: {test_mae:.4f}")

# 7.Epoch
train_mae = np.array(history.history['mae'])
val_mae = np.array(history.history['val_mae'])
epochs_range = range(1, len(val_mae) + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_mae, label='Train MAE')
plt.plot(epochs_range, val_mae, label='Validation MAE')
plt.plot(epochs_range, test_callback.test_mae, label='Test MAE (dynamic)', linestyle='--', color='red')

plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Train / Validation / Test MAE over Epochs')
plt.legend()
plt.grid(True)
plt.show()