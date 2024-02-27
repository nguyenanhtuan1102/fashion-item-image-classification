from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras import layers

# load data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Set label
class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
class_nums = len(class_name)

# Visualize random image
def plot_data(X_data: np.ndarray, y_data: np.ndarray) -> None:
  nrows, ncols = 2, 4
  fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4))
  len_x = X_data.shape[0]
  for idx in range(nrows*ncols):
    ax = axes[idx // ncols, idx % ncols]

    img_idx = random.randint(0, len_x)
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel(class_name[y_data[img_idx]], color="black")

    ax.imshow(X_data[img_idx], cmap="gray")
plot_data(X_train, y_train)

# Encoding dataset
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Add chanel 1 to X_train and X_test
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Category target
y_train_label = keras.utils.to_categorical(y_train, class_nums)
y_test_label = keras.utils.to_categorical(y_test, class_nums)

# Train model
input_shape = (28, 28, 1)
model = keras.models.Sequential([
  layers.Flatten(input_shape=input_shape),
  layers.Dense(512, activation="relu"),
  layers.Dense(256, activation="relu"),
  layers.Dense(128, activation="relu"),
  layers.Dense(class_nums, activation="softmax")
])
model.summary()

model.compile(optimizer="sgd",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

epochs = 10
batch_size = 128
history = model.fit(X_train, y_train_label,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1
          )

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test_label)
print("Loss = {0}".format(test_loss))
print("Accuracy = {0}".format(test_acc))

# Making a prediction on new data
n = random.randint(0, 9999)
plt.imshow(X_test[n])
plt.show()

# We use predict() on new data
predicted_value = model.predict(X_test)
print("Handwritten number in the image is = {0}".format(np.argmax(predicted_value[n])))
history.history.keys()

# Graph representing the model's accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("epoch")
plt.ylabel("accuracy/loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Graph representing the model's accuracy and model's loss
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training loss and accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy/loss")
plt.legend(["accuracy", "val_accuracy", "loss", "val_loss"])
plt.show()
