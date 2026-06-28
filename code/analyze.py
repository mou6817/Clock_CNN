import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model" / "mobilenetv2_clock_best.keras"
IMG_PATH = BASE_DIR / "dataset" / "test" / "4-55" / "44.jpg"

RESULT_DIR = BASE_DIR / "result" / "gradcam"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "block_12_add"


def load_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap).numpy()

    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    original_img = cv2.imread(str(img_path))
    original_img = cv2.resize(original_img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

    cv2.imwrite(str(output_path), overlay)


def save_original(img_path, output_path):
    original_img = cv2.imread(str(img_path))
    original_img = cv2.resize(original_img, IMG_SIZE)
    cv2.imwrite(str(output_path), original_img)


def save_heatmap(heatmap, output_path):
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_path), heatmap_color)


model = load_model(MODEL_PATH)

img_array = load_image(IMG_PATH)

preds = model.predict(img_array)
pred_index = np.argmax(preds[0])
confidence = np.max(preds[0])

print("Image:", IMG_PATH)
print("Predicted index:", pred_index)
print(f"Confidence: {confidence:.4f}")

heatmap = make_gradcam_heatmap(
    img_array,
    model,
    LAST_CONV_LAYER_NAME,
    pred_index=pred_index
)

save_original(
    IMG_PATH,
    RESULT_DIR / "original.png"
)

save_heatmap(
    heatmap,
    RESULT_DIR / "heatmap.png"
)

save_gradcam(
    IMG_PATH,
    heatmap,
    RESULT_DIR / "overlay.png"
)

print("Grad-CAM images saved to:", RESULT_DIR)