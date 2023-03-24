import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

# List available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

# If you have multiple GPUs, you can select which GPU to use by its index (0, 1, 2, etc.)
gpu_index = 0

# If GPUs are available, use the chosen GPU; otherwise, use the CPU
if len(physical_devices) > 0:
    device = f'/device:GPU:{gpu_index}'  # Use the selected gpu
    print(f'Using GPU {gpu_index}\n')
else:
    device = '/device:CPU:0'  # Use the CPU
    print('Using the CPU\n')

# Use the selected device for the following operations
with tf.device(device): 


    np.random.seed(22)

    # Load normal data
    def load_normal(norm_path):
        norm_files = os.listdir(norm_path)
        norm_labels = np.array(['normal']*len(norm_files))

        norm_images = []

        for image in tqdm(norm_files):
            img = cv2.imread(norm_path + image)
            img = cv2.resize(img, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            norm_images.append(img)

        norm_images = np.array(norm_images)

        return norm_images, norm_labels

    # Load pneumonia data
    def load_pneumonia(pneu_path):
        pneu_files = os.listdir(pneu_path)
        pneu_labels = np.array(['pneumonia']*len(pneu_files))

        pneu_images = []

        for image in tqdm(pneu_files):
            img = cv2.imread(pneu_path + image)
            img = cv2.resize(img, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pneu_images.append(img)

        pneu_images = np.array(pneu_images)

        return pneu_images, pneu_labels

    print('Loading data...')

    # All images are stored in _images, all labels are in _labels
    norm_images, norm_labels = load_normal('chest_xray/train/NORMAL/')
    pneu_images, pneu_labels = load_pneumonia('chest_xray/train/PNEUMONIA/')

    # Put all train images to x_train, all train labels to y_train
    x_train = np.append(norm_images, pneu_images, axis=0)
    y_train = np.append(norm_labels, pneu_labels)

    print(x_train.shape)
    print(y_train.shape)

    # Finding out the number of samples of each class
    print(np.unique(y_train, return_counts=True))

    print('Display several images...')
    fig, axes = plt.subplots(ncols=7, nrows=2, figsize=(16, 4))

    indices = np.random.choice(len(x_train), size=14)
    counter = 0

    for i in range(2):
        for j in range(7):
            axes[i,j].set_title(y_train[indices[counter]])
            axes[i,j].imshow(x_train[indices[counter]], cmap='gray')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
            counter += 1
    plt.show()

    print('Loading test images...')
    # Repeat what we did on train data
    norm_images, norm_labels = load_normal('chest_xray/test/NORMAL/')
    pneu_images, pneu_labels = load_pneumonia('chest_xray/test/PNEUMONIA/')
    x_test = np.append(norm_images, pneu_images, axis=0)
    y_test = np.append(norm_labels, pneu_labels)

    # Save the loaded images to pickle files for later use
    with open('pneumonia_images.pickle', 'wb') as f:
        pickle.dump((x_train, y_train, x_test, y_test), f)

    # Load the saved pickle files
    with open('pneumonia_images.pickle', 'rb') as f:
        x_train, y_train, x_test, y_test = pickle.load(f)

    print('Preprocessing data...')

    # Creating a new axis on all y data
    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    # Initialize OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)

    # Converting all labels to one-hot vectors
    y_train_one_hot = one_hot_encoder.fit_transform(y_train)
    y_test_one_hot = one_hot_encoder.transform(y_test)

    print('Reshaping data into 4D tensors...')
    # Reshape the data into(no of samples, height, width, 1), where 1 represents the number of channels
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    print('Data augmentation...')
    # Generate new immages with some randomness
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    datagen.fit(x_train)
    train_gen = datagen.flow(x_train, y_train_one_hot, batch_size=32)

    print('CNN')
    # Define the input shape of the neural network

    input_shape = (x_train.shape[1], x_train.shape[2], 1)

    print('Input shape: ', input_shape)

    input1 = Input(shape=input_shape)

    cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1), 
        padding='same')(input1)
    cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1), 
        padding='same')(cnn)
    cnn = MaxPool2D((2, 2))(cnn)

    cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1), 
        padding='same')(cnn)
    cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1), 
        padding='same')(cnn)
    cnn = MaxPool2D((2, 2))(cnn)

    cnn = Flatten()(cnn)
    cnn = Dense(100, activation='relu')(cnn)
    cnn = Dense(50, activation='relu')(cnn)
    output1 = Dense(2, activation='softmax')(cnn)

    model = Model(inputs=input1, outputs=output1)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Using fit_generator() instead of fit() because we are going to use data augmentation. Note that the randomness is changed every epoch
    history = model.fit_generator(train_gen, epochs=20, validation_data=(x_test, y_test_one_hot))

    # Saving the model
    model.save('pneumonia_cnn.h5')

    print('Displaying accuracy...')
    plt.figure(figsize=(8, 6))
    plt.title('accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()

    print('Displaying loss...')
    plt.figure(figsize=(8, 6))
    plt.title('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()

    # Predicting the test data
    predictions = model.predict(x_test)
    print(predictions)

    predictions = one_hot_encoder.inverse_transform(predictions)

    print('Model evaluation...')
    print(one_hot_encoder.categories_)

    classnames = ['normal', 'pneumonia']

    print('Displaying confusion matrix...')
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 8))
    plt.title('Confusion matrix')
    sns.heatmap(cm, cbar=False, xticklabels=classnames, yticklabels=classnames, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()