from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Bidirectional
from keras.models import Model
from keras.layers import BatchNormalization
from keras import backend as tf_keras_backend
from keras.layers import MaxPool2D
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
from consts import char_list
import numpy as np
import Levenshtein as lv
import matplotlib.pyplot as plt
from keras.models import load_model

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()


def ModelCNNLSTM():
    #shape of height=32 and width=128
    inputs = Input(shape=(32, 128, 1))   #arhitectura modelului preluata

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

    squeezed = Lambda(lambda x: tf_keras_backend.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

    act_model = Model(inputs, outputs)

    return act_model, outputs, inputs


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf_keras_backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train_model(inputs, the_labels, input_length, label_length, outputs, train_data, valid_data, batch_size=5,
                epochs=25, optimizer_name='sgd'):
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])
    model = Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)

    # unpack training and validation data
    train_images, train_padded_label, train_input_length, train_label_length = train_data
    valid_images, valid_padded_label, valid_input_length, valid_label_length = valid_data

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer_name, metrics=['accuracy'])

    # Format file path for saving model checkpoints
    filepath = "{}o-{}e-{}t-{}v.hdf5".format(
        optimizer_name,
        str(epochs),
        str(train_images.shape[0]),
        str(valid_images.shape[0])
    )

    # Define checkpoint to save the best model
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    # Fit the model
    history = model.fit(
        x=[train_images, train_padded_label, train_input_length, train_label_length],
        y=np.zeros(len(train_images)),  # Dummy target since CTC doesn't use labels directly
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            [valid_images, valid_padded_label, valid_input_length, valid_label_length], np.zeros(len(valid_images))),
        verbose=1,
        callbacks=callbacks_list
    )

    #save the model after training
    model.save(filepath='./model3.h5', overwrite=False, include_optimizer=True)

    return history, model


def reload_and_retrain(model_path, train_data, valid_data, batch_size=5, epochs=10):
    """
    Reload a previously saved model and retrain it on new data.

    Args:
        model_path (str): Path to the saved model file (e.g., './model3.h5').
        train_data (tuple): A tuple containing training images, labels, input lengths, and label lengths.
        valid_data (tuple): A tuple containing validation images, labels, input lengths, and label lengths.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.

    Returns:
        history: Training history containing the loss and accuracy.
        model: The retrained model.
    """

    # Load the saved model
    model = load_model(model_path, compile=False)

    # Unpack the new training and validation data
    train_images, train_padded_label, train_input_length, train_label_length = train_data
    valid_images, valid_padded_label, valid_input_length, valid_label_length = valid_data

    # Compile the model again with the same optimizer and loss (CTC loss in this case)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model.optimizer, metrics=['accuracy'])

    # Define a new checkpoint for saving the best model after retraining
    filepath = "{}-retrained-{}e-{}t-{}v.hdf5".format(
        model.optimizer.__class__.__name__.lower(),
        str(epochs),
        str(train_images.shape[0]),
        str(valid_images.shape[0])
    )

    # Save the best model during retraining
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    # Retrain the model on new data
    history = model.fit(
        x=[train_images, train_padded_label, train_input_length, train_label_length],
        y=np.zeros(len(train_images)),  # Dummy target because CTC loss doesn't use y_true directly
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            [valid_images, valid_padded_label, valid_input_length, valid_label_length], np.zeros(len(valid_images))
        ),
        verbose=1,
        callbacks=callbacks_list
    )

    # Save the retrained model
    model.save(filepath='./retrained_model.h5', overwrite=True, include_optimizer=True)

    return history, model


def predict_and_evaluate_model(act_model, filepath, valid_images, valid_original_text, char_list):
    # Load the saved best model weights
    act_model.load_weights(filepath)

    # Predict outputs on validation images
    prediction = act_model.predict(valid_images)

    # Use CTC decoder
    decoded = tf_keras_backend.ctc_decode(prediction,
                          input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                          greedy=True)[0][0]
    out = tf_keras_backend.get_value(decoded)

    # Calculate Jaro distance for predictions
    total_jaro = 0
    for i, x in enumerate(out):
        letters = ''
        for p in x:
            if int(p) != -1:
                letters += char_list[int(p)]
        total_jaro += lv.jaro(letters, valid_original_text[i])

    avg_jaro = total_jaro / len(out)
    print('Average Jaro Similarity:', avg_jaro)

    return out, avg_jaro


def visualize_predictions(act_model, valid_images, valid_original_text, char_list, start_idx=2000, end_idx=2004):
    # Predict outputs on selected validation images
    prediction = act_model.predict(valid_images[start_idx:end_idx])

    # Use CTC decoder
    decoded = tf_keras_backend.ctc_decode(prediction,
                          input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                          greedy=True)[0][0]
    out = tf_keras_backend.get_value(decoded)

    # Display predictions and original text
    idx = start_idx
    for i, x in enumerate(out):
        print("Original Text  :", valid_original_text[idx])
        print("Predicted Text :", end='')
        for p in x:
            if int(p) != -1:
                print(char_list[int(p)], end='')
        print('\n')

        # Show corresponding image
        plt.imshow(valid_images[idx].reshape(32, 128), cmap=plt.cm.gray)
        plt.show()

        idx += 1
