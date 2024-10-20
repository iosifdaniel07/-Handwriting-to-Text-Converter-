import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import Bidirectional
from keras.models import Model
from keras.layers import BatchNormalization
from keras import backend as K
import numpy as np

# Initialize batch size (you can adjust this as needed)
batch_size = 32

# Define CTC loss function
def ctc_loss(y_true, y_pred):
    y_pred = y_pred[:, :, :]
    input_length = np.ones((batch_size, 1)) * y_pred.shape[1]
    label_length = np.ones((batch_size, 1)) * len(y_true[0])
    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Input size based on image dimensions (height, width, channels)
img_height = 128
img_width = 64
num_channels = 1 # Grayscale images
num_classes = 30 # Number of unique characters (you can adjust this based on the dataset)

# CNN Feature Extractor
cnn_input = Input(shape=(img_height, img_width, num_channels))

x = Conv2D(32, (3, 3), padding="same", activation="relu")(cnn_input)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Now, the output shape of x is (batch_size, new_height, new_width, num_filters)
# Reshape for the LSTM input, the shape should be (batch_size, time_steps, features)
# We'll flatten the spatial dimensions (height and width) into a sequence (time_steps)
# time_steps will be the new_width, and features will be new_height * num_filters
new_shape = (img_width // 16, (img_height // 16) * 256)  # new_height and new_width reduced by pooling
x = Reshape(target_shape=new_shape)(x)

# Add batch normalization for regularization
x = BatchNormalization()(x)

# LSTM Network for Sequence Modeling
lstm_units = 256  # You can adjust the number of units here for complexity
x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)

# Dense layer to classify characters (including blank character for CTC)
x = Dense(num_classes + 1, activation="softmax")(x)

# Define the model
model = Model(inputs=cnn_input, outputs=x)

# Compile with Adam optimizer and CTC loss
model.compile(optimizer="adam", loss=ctc_loss)

# Print model summary
model.summary()