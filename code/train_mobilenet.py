import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "model"
RESULT_DIR = BASE_DIR / "result"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


train_dir = "/home/mou/Clock_CNN/dataset/train"
val_dir = "/home/mou/Clock_CNN/dataset/valid"

img_size = (224, 224)
batch_size = 32
num_classes = 144

train_gen = ImageDataGenerator( rescale = 1./255, rotation_range = 15, 
                                zoom_range = 0.1, width_shift_range = 0.1, 
                                height_shift_range = 0.1)

val_gen = ImageDataGenerator(rescale=1./255) 

train_data = train_gen.flow_from_directory(train_dir, target_size = img_size,
                                           batch_size = batch_size, class_mode = 'categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size = img_size,
                                       batch_size = batch_size, class_mode = 'categorical', shuffle=False)

base_model = MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
x =  Dropout(0.3)(x)
output = Dense(num_classes, activation = "softmax")(x)

model = Model(inputs = base_model.input, outputs = output)

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint("model/mobilenetv2_clock_best.keras",
                             monitor = "val_accuracy", save_best_only = True,
                             mode = "max", verbose = 1) # 把Accuracy最高的模型存進去

history = model.fit(train_data, validation_data = val_data, epochs = 20, callbacks = [checkpoint])

print("\n============== Final Epoch ==================")
print(f"Train Accuracy : {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy : {history.history['val_accuracy'][-1]:.4f}")
print(f"Train Loss : {history.history['loss'][-1]:.4f}")
print(f"Validation Loss : {history.history['val_loss'][-1]:.4f}")

best_epoch = np.argmax(history.history["val_accuracy"])

print("\n========== Best Model ==========")
print(f"Best Epoch : {best_epoch + 1}")
print(f"Train Accuracy : {history.history['accuracy'][best_epoch]:.4f}")
print(f"Validation Accuracy : {history.history['val_accuracy'][best_epoch]:.4f}")
print(f"Train Loss : {history.history['loss'][best_epoch]:.4f}")
print(f"Validation Loss : {history.history['val_loss'][best_epoch]:.4f}")


# Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "accuracy.png", dpi=300)
plt.close()

# Loss
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "loss.png", dpi=300)
plt.close()