import cv2
import numpy as np

from consts import char_list


def read_words_from_file(file_path, start_line=1, end_lines=22539):
    if end_lines > 115338:
        end_lines = 115338
        start_line = start_line + 18# check words doc to see why + 18...
    with open(file_path, 'r') as f:
        contents = f.readlines()[start_line:end_lines]
    lines = [line.strip() for line in contents]
    return lines


def process_image(img): # preluat
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32 - w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128 - h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # normalize
    img = img / 255

    return img


def encode_to_labels(txt):
    # encode cuvinte in cifre
    dig = []
    for index, chara in enumerate(txt):
        dig.append(char_list.index(chara))

    return dig


def process_lines(lines, process_image, encode_to_labels):
    train_images, train_labels, train_input_length, train_label_length, train_original_text = [], [], [], [], []
    valid_images, valid_labels, valid_input_length, valid_label_length, valid_original_text = [], [], [], [], []
    max_label_len = 0

    # Calculate the split for training and validation
    train_count = int(0.75 * len(lines))  # 75% for training

    for index, line in enumerate(lines):
        splits = line.split(' ')
        status = splits[1]

        if status == 'ok':
            word_id = splits[0]
            word = "".join(splits[8:])

            splits_id = word_id.split('-')
            filepath = 'words/{}/{}-{}/{}.png'.format(splits_id[0], splits_id[0], splits_id[1], word_id)

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            try:
                img = process_image(img)
            except Exception as e:
                print(f"Error processing image {filepath}: {e}")
                continue

            try:
                label = encode_to_labels(word)
            except Exception as e:
                print(f"Error encoding label for word {word}: {e}")
                continue

            # Determine if the current index goes to training or validation
            if index < train_count:
                train_images.append(img)
                train_labels.append(label)
                train_input_length.append(31)
                train_label_length.append(len(word))
                train_original_text.append(word)
            else:
                valid_images.append(img)
                valid_labels.append(label)
                valid_input_length.append(31)
                valid_label_length.append(len(word))
                valid_original_text.append(word)

            if len(word) > max_label_len:
                max_label_len = len(word)

    return {
        'train': {
            'images': train_images,
            'labels': train_labels,
            'input_length': train_input_length,
            'label_length': train_label_length,
            'original_text': train_original_text
        },
        'valid': {
            'images': valid_images,
            'labels': valid_labels,
            'input_length': valid_input_length,
            'label_length': valid_label_length,
            'original_text': valid_original_text
        },
        'max_label_len': max_label_len
    }
