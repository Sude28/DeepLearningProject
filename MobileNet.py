import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 📁 1. Klasör yolları
base_dir = r"D:\DERİN_PROJE2\DERİN_PROJE2\dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "test")  # test klasörü doğrulama için kullanılıyor


# 📦 2. Görüntü verisi ayarları
batch_size = 32
image_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())

# 🧠 3. MobileNet tabanlı model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ⚙️ 4. Derleme
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ⏱️ Eğitim süresi başlangıcı
start_time = time.time()

# 📦 5. Callback tanımı
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("mobilenet_best_model.keras", save_best_only=True)
]

# 🏋️ 6. Eğitim
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

# ⏱️ Eğitim süresi bitişi
end_time = time.time()
print(f"⏱️ Eğitim süresi: {(end_time - start_time)/60:.2f} dakika")

# 💾 7. Eğitim sonrası model kaydı
model.save("mobilenet_final_model.keras")

# 📊 8. Eğitim grafiği
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
for ep in [5, 10, 20]:
    if ep <= len(history.history['val_accuracy']):
        plt.scatter(ep-1, history.history['val_accuracy'][ep-1], color='red')
        plt.text(ep-1, history.history['val_accuracy'][ep-1], f"{history.history['val_accuracy'][ep-1]:.2f}", fontsize=9)
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Doğruluk Grafiği')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
for ep in [5, 10, 20]:
    if ep <= len(history.history['val_loss']):
        plt.scatter(ep-1, history.history['val_loss'][ep-1], color='red')
        plt.text(ep-1, history.history['val_loss'][ep-1], f"{history.history['val_loss'][ep-1]:.2f}", fontsize=9)
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Model Kayıp Grafiği')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mobilenet_training_plot.png")
plt.show()

# 📈 9. Değerlendirme
val_generator.reset()
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_generator.classes

# Sınıflandırma raporu ve karışıklık matrisi
print("📋 Sınıf Etiketleri:", class_labels)
print("\n🧮 Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)
print("🔍 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Ek olarak metrikleri CSV dosyasına kaydet
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [np.mean(history.history['val_accuracy']), precision, recall, f1]
})
results_df.to_csv("mobilenet_metrics.csv", index=False)

