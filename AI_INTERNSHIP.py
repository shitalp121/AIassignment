#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau

#dataset
data = [
    {'user': 'user1', 'purchases': ['amazon', 'flipkart', 'myntra']},
    {'user': 'user2', 'purchases': ['amazon', 'flipkart']},
    {'user': 'user3', 'purchases': ['nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user4', 'purchases': ['snapdeal', 'amazon']},
    {'user': 'user5', 'purchases': ['myntra', 'nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user6', 'purchases': ['amazon', 'snapdeal', 'flipkart']},
    {'user': 'user7', 'purchases': ['flipkart', 'myntra']},
    {'user': 'user8', 'purchases': ['myntra', 'snapdeal']},
    {'user': 'user9', 'purchases': ['swiggy', 'amazon', 'myntra']},
    {'user': 'user10', 'purchases': ['amazon', 'zomato', 'snapdeal']},
    {'user': 'user11', 'purchases': ['flipkart', 'amazon']},
    {'user': 'user12', 'purchases': ['myntra', 'flipkart', 'snapdeal', 'amazon']},
    {'user': 'user13', 'purchases': ['snapdeal', 'myntra']},
    {'user': 'user14', 'purchases': ['swiggy', 'myntra', 'flipkart']},
    {'user': 'user15', 'purchases': ['zomato', 'snapdeal', 'amazon']},
    {'user': 'user16', 'purchases': ['snapdeal', 'amazon', 'flipkart']},
    {'user': 'user17', 'purchases': ['amazon', 'myntra']},
    {'user': 'user18', 'purchases': ['flipkart', 'snapdeal']},
    {'user': 'user19', 'purchases': ['myntra', 'flipkart', 'amazon']},
    {'user': 'user20', 'purchases': ['snapdeal', 'zomato']},
    {'user': 'user21', 'purchases': ['amazon', 'snapdeal', 'flipkart', 'myntra']},
    {'user': 'user22', 'purchases': ['myntra', 'amazon']},
    {'user': 'user23', 'purchases': ['zomato', 'myntra', 'amazon']},
    {'user': 'user24', 'purchases': ['snapdeal', 'flipkart', 'amazon']},
    {'user': 'user25', 'purchases': ['myntra', 'swiggy', 'snapdeal']},
    {'user': 'user26', 'purchases': ['amazon', 'flipkart', 'myntra']},
    {'user': 'user27', 'purchases': ['amazon', 'flipkart']},
    {'user': 'user28', 'purchases': ['nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user29', 'purchases': ['snapdeal', 'amazon']},
    {'user': 'user30', 'purchases': ['myntra', 'nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user31', 'purchases': ['amazon', 'snapdeal', 'flipkart']},
    {'user': 'user32', 'purchases': ['flipkart', 'myntra']},
    {'user': 'user33', 'purchases': ['myntra', 'snapdeal']},
    {'user': 'user34', 'purchases': ['swiggy', 'amazon', 'myntra']},
    {'user': 'user35', 'purchases': ['amazon', 'zomato', 'snapdeal']},
    {'user': 'user36', 'purchases': ['flipkart', 'amazon']},
    {'user': 'user37', 'purchases': ['myntra', 'flipkart', 'snapdeal', 'amazon']},
    {'user': 'user38', 'purchases': ['snapdeal', 'myntra']},
    {'user': 'user39', 'purchases': ['swiggy', 'myntra', 'flipkart']},
    {'user': 'user40', 'purchases': ['zomato', 'snapdeal', 'amazon']},
    {'user': 'user41', 'purchases': ['snapdeal', 'amazon', 'flipkart']},
    {'user': 'user42', 'purchases': ['amazon', 'myntra']},
    {'user': 'user43', 'purchases': ['flipkart', 'snapdeal']},
    {'user': 'user44', 'purchases': ['myntra', 'flipkart', 'amazon']},
    {'user': 'user45', 'purchases': ['snapdeal', 'zomato']},
    {'user': 'user46', 'purchases': ['amazon', 'snapdeal', 'flipkart', 'myntra']},
    {'user': 'user47', 'purchases': ['myntra', 'amazon']},
    {'user': 'user48', 'purchases': ['zomato', 'myntra', 'amazon']},
    {'user': 'user49', 'purchases': ['zomato', 'myntra', 'amazon']},
    {'user': 'user50', 'purchases': ['snapdeal', 'flipkart', 'amazon']},
    {'user': 'user51', 'purchases': ['amazon', 'flipkart', 'myntra']},
    {'user': 'user52', 'purchases': ['amazon', 'flipkart']},
    {'user': 'user53', 'purchases': ['nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user54', 'purchases': ['snapdeal', 'amazon']},
    {'user': 'user55', 'purchases': ['myntra', 'nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user56', 'purchases': ['amazon', 'snapdeal', 'flipkart']},
    {'user': 'user57', 'purchases': ['flipkart', 'myntra']},
    {'user': 'user58', 'purchases': ['myntra', 'snapdeal']},
    {'user': 'user59', 'purchases': ['swiggy', 'amazon', 'myntra']},
    {'user': 'user60', 'purchases': ['amazon', 'zomato', 'snapdeal']},
    {'user': 'user61', 'purchases': ['flipkart', 'amazon']},
    {'user': 'user62', 'purchases': ['myntra', 'flipkart', 'snapdeal', 'amazon']},
    {'user': 'user63', 'purchases': ['snapdeal', 'myntra']},
    {'user': 'user64', 'purchases': ['swiggy', 'myntra', 'flipkart']},
    {'user': 'user65', 'purchases': ['zomato', 'snapdeal', 'amazon']},
    {'user': 'user66', 'purchases': ['snapdeal', 'amazon', 'flipkart']},
    {'user': 'user67', 'purchases': ['amazon', 'myntra']},
    {'user': 'user68', 'purchases': ['flipkart', 'snapdeal']},
    {'user': 'user69', 'purchases': ['myntra', 'flipkart', 'amazon']},
    {'user': 'user70', 'purchases': ['snapdeal', 'zomato']},
    {'user': 'user71', 'purchases': ['amazon', 'snapdeal', 'flipkart', 'myntra']},
    {'user': 'user72', 'purchases': ['myntra', 'amazon']},
    {'user': 'user73', 'purchases': ['zomato', 'myntra', 'amazon']},
    {'user': 'user74', 'purchases': ['snapdeal', 'flipkart', 'amazon']},
    {'user': 'user75', 'purchases': ['myntra', 'swiggy', 'snapdeal']},
    {'user': 'user76', 'purchases': ['amazon', 'flipkart', 'myntra']},
    {'user': 'user77', 'purchases': ['amazon', 'flipkart']},
    {'user': 'user78', 'purchases': ['nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user79', 'purchases': ['snapdeal', 'amazon']},
    {'user': 'user80', 'purchases': ['myntra', 'nykaa', 'flipkart', 'snapdeal']},
    {'user': 'user81', 'purchases': ['amazon', 'snapdeal', 'flipkart']},
    {'user': 'user82', 'purchases': ['flipkart', 'myntra']},
    {'user': 'user83', 'purchases': ['myntra', 'snapdeal']},
    {'user': 'user84', 'purchases': ['swiggy', 'amazon', 'myntra']},
    {'user': 'user85', 'purchases': ['amazon', 'zomato', 'snapdeal']},
    {'user': 'user86', 'purchases': ['flipkart', 'amazon']},
    {'user': 'user87', 'purchases': ['myntra', 'flipkart', 'snapdeal', 'amazon']},
    {'user': 'user88', 'purchases': ['snapdeal', 'myntra']},
    {'user': 'user89', 'purchases': ['swiggy', 'myntra', 'flipkart']},
    {'user': 'user90', 'purchases': ['zomato', 'snapdeal', 'amazon']},
    {'user': 'user91', 'purchases': ['snapdeal', 'amazon', 'flipkart']},
    {'user': 'user92', 'purchases': ['amazon', 'myntra']},
    {'user': 'user93', 'purchases': ['flipkart', 'snapdeal']},
    {'user': 'user94', 'purchases': ['myntra', 'flipkart', 'amazon']},
    {'user': 'user95', 'purchases': ['snapdeal', 'zomato']},
    {'user': 'user96', 'purchases': ['amazon', 'snapdeal', 'flipkart', 'myntra']},
    {'user': 'user97', 'purchases': ['myntra', 'amazon']},
    {'user': 'user98', 'purchases': ['zomato', 'myntra', 'amazon']},
    {'user': 'user99', 'purchases': ['snapdeal', 'flipkart', 'amazon']},
    {'user': 'user100', 'purchases': ['snapdeal', 'zomato', 'amazon']},
    
]

# Create a mapping from platforms to indices
platforms = list(set([item for sublist in [d['purchases'] for d in data] for item in sublist]))
platform_to_index = {platform: idx for idx, platform in enumerate(platforms)}
index_to_platform = {idx: platform for platform, idx in platform_to_index.items()}

# Convert purchases to sequences of indices
sequences = [[platform_to_index[p] for p in d['purchases']] for d in data]

# Prepare training data
X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])
        
# Pad sequences for consistent input shape
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='pre')
y = np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


# Define the model with additional improvements
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(platform_to_index), output_dim=256, input_length=X_train.shape[1]),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(len(platform_to_index), activation='softmax')
])

# Compile the model with a different learning rate and optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate reduction callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model with early stopping and learning rate scheduler
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions
predictions = model.predict(X_test)
predicted_indices = np.argmax(predictions, axis=1)
predicted_platforms = [index_to_platform[idx] for idx in predicted_indices]

# Print some sample predictions
for i in range(5):
    print(f"True: {index_to_platform[y_test[i]]}, Predicted: {predicted_platforms[i]}")

