import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Define 
model = keras.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def train() :
 model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
 model.save('my_minsty_boy.h5')

def evaluate():
 test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
 print(f"Test accuracy: {test_acc * 100:.2f}%")

def test():
    img = Image.open("test.png").convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match MNIST

    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0

# Invert colors if needed (MNIST digits are white on black)
    img_array = 1 - img_array

# Reshape for model input
    img_array = img_array.reshape(1, 28, 28)

# Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    print(f"Predicted Label: {predicted_label}")
