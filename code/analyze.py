from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
                                       batch_size = batch_size, class_mode = 'categorical',
                                       shuffle=False) # 要加shuffle不然圖片順序會被打亂


model = load_model("model/mobilenetv2_clock_stage1.keras")
pred = model.predict(val_data)
pred_index = np.argmax(pred, axis = 1)
true_index = val_data.classes

loss, acc = model.evaluate(val_data)
print("evaluate acc:", acc)

index_to_class = {v:k for k, v in val_data.class_indices.items()}

# for i in range(20):
#     true_label = index_to_class[true_index[i]]
#     pred_label = index_to_class[pred_index[i]]
#     confidence = np.max(pred[i])

#     print(f"True Label: {true_label}, Predicted Label: {pred_label}, Confidence: {confidence:.4f}")

wrong = np.where(pred_index != true_index)[0]


# for i in wrong:
#     true_label = index_to_class[true_index[i]]
#     pred_label = index_to_class[pred_index[i]]
#     confidence = np.max(pred[i])
#     filename = val_data.filenames[i]

#     print(f"Filename: {filename}, True Label: {true_label}, Predicted Label: {pred_label}, Confidence: {confidence:.4f}")