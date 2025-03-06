# MNIST Digit Classifier

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-green.svg)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-x.x-purple.svg)](https://pillow.readthedocs.io/en/stable/)


This project is a backend-only Python application that trains, evaluates, and tests a Convolutional Neural Network (CNN) model for classifying handwritten digits from the MNIST dataset.  It uses TensorFlow/Keras for model building and training, NumPy for numerical operations, and Pillow for image manipulation.

## Accessible Routes & Methods

The application is run from the command line and doesn't have traditional web routes.  Instead, it offers the following command-line actions:


* **`python main.py train`**:  Trains the MNIST model using the MNIST dataset.  No parameters are required. This command saves the model to `mnist_model.h5`

* **`python main.py evaluate`**: Loads the model from `mnist_model.h5` and evaluates its performance on the MNIST test dataset. Prints the test accuracy. No parameters are required.

* **`python main.py test`**: Loads the model from `mnist_model.h5` and uses it to predict the digit in `test.png`.  Requires a `test.png` image in the same directory.  The image should be a 28x28 grayscale image of a handwritten digit.

* **`python main.py easy`**: Loads the model from `mnist_model.h5` and uses it to predict the digit in `test_easy.png`. Requires a `test_easy.png` image in the same directory.  The image should be a 28x28 grayscale image of a handwritten digit.


All actions use the `GET` method implicitly through command-line arguments.  No parameters are passed directly as part of a request, but the presence or absence of specific files (e.g., `test.png`) indirectly influences the behavior.  Error handling is implemented to gracefully handle the absence of a trained model or required image files.
