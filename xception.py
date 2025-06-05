import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

base_dir = r"D:\DERİN_PROJE2\DERİN_PROJE2\dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "test")

batch_size = 32
image_size = (299, 299)

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

base_model = Xception(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath="Xception_epoch_{epoch:02d}.keras", save_freq='epoch', save_best_only=False)
]

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

end_time = time.time()
print(f"⏱️ Eğitim süresi: {(end_time - start_time)/60:.2f} dakika")

model.save("Xception_final_model.keras")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
for ep in [5, 10, 20]:
    if ep <= len(history.history['val_accuracy']):
        plt.scatter(ep-1, history.history['val_accuracy'][ep-1], color='red')
        plt.text(ep-1, history.history['val_accuracy'][ep-1], f"{{history.history['val_accuracy'][ep-1]:.2f}}", fontsize=9)
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Xception Doğruluk Grafiği')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
for ep in [5, 10, 20]:
    if ep <= len(history.history['val_loss']):
        plt.scatter(ep-1, history.history['val_loss'][ep-1], color='red')
        plt.text(ep-1, history.history['val_loss'][ep-1], f"{{history.history['val_loss'][ep-1]:.2f}}", fontsize=9)
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Xception Kayıp Grafiği')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Xception_training_plot.png")
plt.show()

val_generator.reset()
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_generator.classes

print("\n🧮 Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)
print("🔍 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
results_df = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            'Value': [np.mean(history.history['val_accuracy']), precision, recall, f1]})
results_df.to_csv("Xception_metrics.csv", index=False)
