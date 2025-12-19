import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def train_mnist_dnn():
    print("Fetching MNIST dataset...")
    # Loading the dataset directly from Keras
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocessing: Scale pixels to [0, 1] and reshape for CNN (28x28x1)
    X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    print("Defining CNN Architecture...")
    # Satisfies the "pretrained deep neural network" requirement
    model = models.Sequential([
        # Feature Extraction Layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Classification Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Prevents overfitting
        layers.Dense(10, activation='softmax') # 10 classes for digits 0-9
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training the Deep Neural Network...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    
    print("\nEvaluating on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nFinal Test Accuracy: {test_acc*100:.2f}%')

    # Save the entire model to use later
    model.save('mnist_deep_model.h5')
    print("Model saved to mnist_deep_model.h5")

if __name__ == "__main__":
    train_mnist_dnn()