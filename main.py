import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Define model
model = keras.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])


# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def train():
    """Train and save the model."""
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
    model.save("mnist_model.h5")  # Save the trained model
    print("Model trained and saved as mnist_model.h5")

def evaluate():
    """Evaluate the model on the test dataset."""
    model.load_weights("mnist_model.h5")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

def test():
    """Test on a custom image."""
    try:
        model.load_weights("mnist_model.h5")
        print("Loaded trained model.")
    except:
        print("No trained model found. Train the model first.")
        return

    img = Image.open("test.png").convert("L")   
    img = img.resize((28, 28))  # Resize to match MNIST

 
    img_array = np.array(img) / 255.0
 
    img_array = 1 - img_array

 
    img_array = img_array.reshape(1, 28, 28, 1)

   
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    print(f"Predicted Label: {predicted_label}")

def test_easy():
    """Test on a custom easy image."""
    try:
        model.load_weights("mnist_model.h5")
        print("Loaded trained model.")
    except:
        print("No trained model found. Train the model first.")
        return

    img = Image.open("test_easy.png").convert("L")   
    img = img.resize((28, 28))  # Resize to match MNIST

 
    img_array = np.array(img) / 255.0
 
    img_array = 1 - img_array

 
    img_array = img_array.reshape(1, 28, 28, 1)

   
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    print(f"Predicted Label: {predicted_label}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate, or test the MNIST model.")
    parser.add_argument("action", choices=["train", "evaluate", "test", "easy"], help="Action to perform")

    args = parser.parse_args()

    if args.action == "train":
        train()
    elif args.action == "evaluate":
        evaluate()
    elif args.action == "test":
        test()
    elif args.action == "easy":
        test_easy()
