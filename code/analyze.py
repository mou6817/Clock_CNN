import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path



model_path = "/home/mou/Clock_CNN/model/mobilenetv2_clock_best.keras"
img_path = "/home/mou/Clock_CNN/dataset/test/4-55/44.jpg"

img_size = (224, 224)
last_conv_layer_name = "out_relu"

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)

model = load_model(model_path)

def load_img_array(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

img_array = load_img_array(img_path)

preds = model.predict(img_array)
pred_index = np.argmax(preds[0])
confidence = np.max(preds[0])

print("Predicted index:", pred_index)
print("Confidence:", confidence)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.output
    ]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_channel = predictions[:, pred_index]

grads = tape.gradient(class_channel, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
original_img = cv2.resize(original_img, img_size)

heatmap_resized = cv2.resize(heatmap, img_size)
heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

superimposed_img = heatmap_color * 0.4 + original_img
superimposed_img = np.uint8(superimposed_img)

plt.figure(figsize=(5,5))
plt.imshow(original_img)
plt.axis("off")
plt.savefig(RESULT_DIR / "original_img.png",
    bbox_inches="tight", pad_inches=0)
plt.close()

plt.figure(figsize=(5,5))
plt.imshow(heatmap_resized, cmap="jet")
plt.axis("off")
plt.savefig(RESULT_DIR / "gradcam.png",
    bbox_inches="tight", pad_inches=0)
plt.close()

plt.figure(figsize=(5,5))
plt.imshow(superimposed_img)
plt.axis("off")
plt.savefig(RESULT_DIR / "overlay.png",
    bbox_inches="tight", pad_inches=0)
plt.close()
