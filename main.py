from consts import RECORDS_COUNT, char_list
from model import Model1, predict_and_evaluate_model, visualize_predictions
from utils import read_words_from_file, process_lines, process_image, encode_to_labels
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Input

file_path = 'words.txt'
lines = read_words_from_file(file_path)

result = process_lines(lines, RECORDS_COUNT, process_image, encode_to_labels)

# Access the returned values
train_data = result['train']
valid_data = result['valid']
max_label_len = result['max_label_len']

train_padded_label = pad_sequences(train_data["labels"],
                                   maxlen=max_label_len,
                                   padding='post',
                                   value=len(char_list))

# Similarly, for validation data (if needed)
valid_padded_label = pad_sequences(valid_data["labels"],
                                   maxlen=max_label_len,
                                   padding='post',
                                   value=len(char_list))

train_images = np.asarray(train_data["images"])
train_input_length = np.asarray(train_data["input_length"])
train_label_length = np.asarray(train_data["label_length"])

valid_images = np.asarray(valid_data["images"])
valid_input_length = np.asarray(valid_data["input_length"])
valid_label_length = np.asarray(valid_data["label_length"])

act_model, outputs, inputs = Model1()
act_model.summary()

the_labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# history, trained_model = train_model(
#     inputs=inputs,
#     the_labels=the_labels,
#     input_length=input_length,
#     label_length=label_length,
#     outputs=outputs,
#     train_data=(train_images, train_padded_label, train_input_length, train_label_length),
#     valid_data=(valid_images, valid_padded_label, valid_input_length, valid_label_length),
#     batch_size=5,
#     epochs=25,
#     optimizer_name='sgd',
#     RECORDS_COUNT=1000
# )

out, avg_jaro = predict_and_evaluate_model(act_model, './sgdo-30000r-25e-18074t-2007v.hdf5', valid_images, valid_data['original_text'], char_list)

visualize_predictions(act_model, valid_images, valid_data['original_text'], char_list, start_idx=2000, end_idx=2004)

#plot_graph(history)

#best_loss, best_acc, best_val_loss, best_val_acc = get_best_model_info(history)
