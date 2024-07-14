# FASHION ITEM IMAGES CLASSIFICATION

This repository contains Python code that utilizes ```TensorFlow``` and ```Keras``` to build a machine learning model for classifying images from the Fashion-MNIST dataset. The model identifies various clothing items such as t-shirts, trousers, dresses, and more.

![fashion-mnist](https://github.com/user-attachments/assets/e33a2a52-73a6-48dd-ae90-13cdc89a397a)

# Dependencies:

- ```tensorflow```
- ```keras```
- ```matplotlib```
- ```numpy```

# Explanation of the Code:

### 1. Imports:

Necessary libraries are imported for working with data, building the model, and visualization.

``` bash
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras import layers
```

### 2. Load data and set labels:

- The pre-loaded Fashion-MNIST dataset is downloaded using ```keras``` utilities.
- The dataset is split into training and testing sets, with ```X``` representing the image data and ```y``` representing the corresponding labels.

- This line loads the pre-existing Fashion-MNIST dataset using ```keras``` utilities.
- The data is split into two sets:
  - Training set ```X_train```, ```y_train```: Used to train the model.
  - Testing set ```X_test```, ```y_test```: Used to evaluate the model's performance after training.
- ```X``` represents the image data, while ```y``` represents the corresponding labels (categories of clothing items).

``` bash
# load data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
```

- class_name: A list containing the names of the ten clothing categories in the dataset.
- class_nums: The number of classes (categories), which is 10 in this case.

``` bash
# Set label
class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
class_nums = len(class_name)
```

### 3. Visualize data:

The plot_data function displays a grid of random images with their associated class labels.

``` bash
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
```

![13](https://github.com/tuanng1102/classify-image-with-neural-network-on-fashion-mnist-dataset/assets/147653892/2ee20c8a-8dee-4476-a3b2-8782075738ab)

### 4. Preprocessing data:

This section involves several data preparation steps:
- Normalization:
    - Converts the image data from unsigned integers (0-255) to floating-point values (0-1) using ```astype(np.float32)``` and division by 255. This normalization helps improve the training process.
- Adding Channel Dimension:
    - Reshapes the data to add a channel dimension (typically representing colors in RGB images) even though the Fashion-MNIST dataset is grayscale. This is achieved using np.expand_dims(axis=-1).
- One-Hot Encoding:
    - Converts the class labels (integers representing categories) into one-hot encoded vectors using ```keras.utils.to_categorical```. This is necessary because the model uses ```categorical-crossentropy``` loss, which assumes the labels are probability distributions.

``` bash
# Encoding dataset
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Add chanel 1 to X_train and X_test
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Category target
y_train_label = keras.utils.to_categorical(y_train, class_nums)
y_test_label = keras.utils.to_categorical(y_test, class_nums)
```

### 5. Build the model:

- A sequential model is created using ```keras.models.Sequential```.
- Layers in the model:
  - ```Flatten```: Flattens the 2D image data into a 1D vector.
  - ```Dense```: Fully connected layers with 512, 256, and 128 units, each using the ReLU activation function for non-linearity.
  - ```Dense```: Final output layer with the number of units equal to the number of classes (10) and a softmax activation function for probability distribution.
  - The model summary provides information about the layers and their parameters.

``` bash
input_shape = (28, 28, 1)
model = keras.models.Sequential([
  layers.Flatten(input_shape=input_shape),
  layers.Dense(512, activation="relu"),
  layers.Dense(256, activation="relu"),
  layers.Dense(128, activation="relu"),
  layers.Dense(class_nums, activation="softmax")
])
model.summary()
```

![7](https://github.com/tuanng1102/classify-image-with-neural-network-on-fashion-mnist-dataset/assets/147653892/268fc7e9-7052-4234-bd87-9c4c86f98d85)


### 6. Compile the model:

The model is compiled with the ```sgd``` optimizer, ```categorical-crossentropy``` loss function suitable for multi-class classification, and ```accuracy``` metric.

``` bash
model.compile(optimizer="sgd",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
```

### 7. Train the model:

- The model is trained on the training data for 10 ```epochs``` (iterations) with a ```batch_size``` of 128 samples.
- A validation split of 10% is used to monitor performance on unseen data during training.

``` bash
epochs = 10
batch_size = 128
history = model.fit(X_train, y_train_label,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1
)
```

![8](https://github.com/tuanng1102/classify-image-with-neural-network-on-fashion-mnist-dataset/assets/147653892/7dd6d9cb-b69f-4314-9bd1-049a2bcde8f0)

### 8. Evaluate the model:

The final model's performance is evaluated on the testing data using ```model.evaluate```, and the loss and accuracy are printed.

``` bash
test_loss, test_acc = model.evaluate(X_test, y_test_label)
print("Loss = {0}".format(test_loss))
print("Accuracy = {0}".format(test_acc))
```

### 9. Make predictions:

- A random image from the testing set is selected and displayed.
= The model predicts the class of the selected image using ```model.predict```, and the predicted class label is printed.

``` bash
# Making a prediction on new data
n = random.randint(0, 9999)
plt.imshow(X_test[n])
plt.show()

# We use predict() on new data
predicted_value = model.predict(X_test)
print("Handwritten number in the image is = {0}".format(np.argmax(predicted_value[n])))
history.history.keys()
```

![11](https://github.com/tuanng1102/classify-image-with-neural-network-on-fashion-mnist-dataset/assets/147653892/0c5b9a95-d0df-4497-bb62-7b2df268d27c)

``` bash
# result
Handwritten number in the image is = Bag
```

### 10. Visualize training history:

Two plots are created to visualize the model's training and validation accuracy and loss over epochs.

``` bash
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
```

![9](https://github.com/tuanng1102/classify-image-with-neural-network-on-fashion-mnist-dataset/assets/147653892/b3d2d9e4-f49b-47e7-84a8-401981981a81)

![10](https://github.com/tuanng1102/classify-image-with-neural-network-on-fashion-mnist-dataset/assets/147653892/ee59ac17-4ad0-40d9-8cd8-955f9488f7d7)
