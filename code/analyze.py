from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections import Counter
from collections import defaultdict

train_dir = "/home/mou/Clock_CNN/dataset/train"
test_dir = "/home/mou/Clock_CNN/dataset/test"

img_size = (224, 224)
batch_size = 32
num_classes = 144

train_gen = ImageDataGenerator( rescale = 1./255, rotation_range = 15, 
                                zoom_range = 0.1, width_shift_range = 0.1, 
                                height_shift_range = 0.1)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size = img_size,
                                           batch_size = batch_size, class_mode = 'categorical')

test_data = val_gen.flow_from_directory(test_dir, target_size = img_size,
                                       batch_size = batch_size, class_mode = 'categorical',
                                       shuffle=False) # 要加shuffle不然圖片順序會被打亂


model = load_model("model/mobilenetv2_clock_best.keras")
pred = model.predict(test_data)
pred_index = np.argmax(pred, axis = 1)
true_index = test_data.classes


index_to_class = {v:k for k, v in test_data.class_indices.items()}

# for i in range(20):
#     true_label = index_to_class[true_index[i]]
#     pred_label = index_to_class[pred_index[i]]
#     confidence = np.max(pred[i])

#     print(f"True Label: {true_label}, Predicted Label: {pred_label}, Confidence: {confidence:.4f}")

wrong = np.where(pred_index != true_index)[0]

confusion_count = defaultdict(int)
confusion_confidence = defaultdict(list)

for i in wrong:
    true_label = index_to_class[true_index[i]]
    pred_label = index_to_class[pred_index[i]]

    pair = (true_label, pred_label)

    confidence = np.max(pred[i])
    filename = test_data.filenames[i]

    print(f"Filename: {filename}, True Label: {true_label}, Predicted Label: {pred_label}, Confidence: {confidence:.4f}")

    confusion_count[pair] += 1
    confusion_confidence[pair].append(confidence)

from collections import Counter

# 找出所有錯誤案例的「真實類別 -> 預測類別」
wrong_pairs = []

for i in range(len(true_index)):
    if pred_index[i] != true_index[i]:
        true_label = index_to_class[true_index[i]]
        pred_label = index_to_class[pred_index[i]]
        wrong_pairs.append((true_label, pred_label))

# 統計最常出現的混淆組合
confusion_count = Counter(wrong_pairs)

print("\n========== Top 5 Deserved Analysis ==========")

top5 = sorted(confusion_count.items(), key = lambda x:x[1], reverse = True)[:5]

for rank, ((true_label, pred_label), count) in enumerate(top5, start=1):
    avg_confidence = np.mean(confusion_confidence[(true_label, pred_label)])
    print(f"{rank}. True: {true_label} -> Predicted: {pred_label}, Count: {count}, Confidence: {avg_confidence:.4f}")

acc = np.mean(pred_index == true_index)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Total Samples: {len(true_index)}, Wrong Predictions: {len(wrong)}")