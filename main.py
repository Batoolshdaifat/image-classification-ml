import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# تحميل بيانات CIFAR-10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# تطبيع البيانات
train_images = train_images / 255.0
test_images = test_images / 255.0

# أسماء الكلاسات
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# بناء نموذج CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    # تحسين مهم: softmax
    layers.Dense(10, activation='softmax')
])

# compile النموذج
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# تدريب النموذج
history = model.fit(
    train_images, train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

# تقييم النموذج
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy:", test_acc)

# 📊 رسم النتائج (Accuracy)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# 📊 رسم الخسارة (Loss)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Model Loss")
plt.show()
