# AI assignment
This repository contains a deep learning model built using TensorFlow to predict the next user purchase based on historical data. The model leverages RNN and LSTM layers to capture sequential dependencies, enhancing the recommendation system for GenZDealZ.ai.
Deep Learning Model for Predicting User Purchase History
##Overview
This project aims to develop a deep learning model using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks to predict the next purchase a user might make based on their historical purchase data. This model is designed to enhance the recommendation system for the GenZDealZ.ai platform.

##Table of Contents
Project Overview
Dataset
Model Architecture
Training and Evaluation
Results
Challenges
Future Work
Installation
Usage
Contributing
License
Dataset
The dataset consists of simulated user purchase histories, where each entry includes a user ID and a sequence of purchases. The format of the data is as follows:

data = [
    {'user': 'user1', 'purchases': ['amazon', 'flipkart', 'myntra']},
    {'user': 'user2', 'purchases': ['amazon', 'flipkart']},
    # More user data...
]

##Model Architecture
The model uses TensorFlow and includes the following layers:

Embedding Layer: Converts platform indices to dense vectors.
Conv1D Layer: Applies convolutional filters.
MaxPooling Layer: Reduces dimensionality.
Bidirectional LSTM Layers: Captures temporal dependencies from both directions.
Dropout Layers: Prevents overfitting.
Dense Layer: Fully connected layer with ReLU activation.
Batch Normalization: Normalizes activations.
Output Layer: Softmax activation for multi-class classification.

##Training and Evaluation
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. It uses early stopping and learning rate reduction callbacks to prevent overfitting and improve convergence.

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

##Results
The model achieved a test accuracy of 51.43%. Sample predictions were made to compare true and predicted values.

##Challenges
Model Complexity: Balancing complexity and overfitting.
Hyperparameter Tuning: Optimizing learning rate, batch size, and number of epochs.
Future Work
Larger Dataset: Using a more extensive and diverse dataset.
Feature Engineering: Including additional features such as time of purchase, user demographics, etc.
Hyperparameter Optimization: Further tuning of hyperparameters.
