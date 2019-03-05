from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
import os, cv2, re, random
import numpy as np
import matplotlib.pyplot as plt

optimizer = Adam(0.0002, 0.5)
D = None #Sequential model for discriminator
G = None #Sequential model for generator
stacked = None #Sequential model for GAN
batch_size = 2
epochs = 10

#input image dimensions
img_width = 150
img_height = 150
input_shape = (img_width, img_height, 3) #3 channels for color

TRAIN_DIREC = 'dogs-cats/train'
TEST_DIREC = 'dogs-cats/test1'

train = os.listdir(TRAIN_DIREC)[0:25]
#train = ["".join(i.split(".")[0:-1]) + "." + i.split(".")[-1] for i in train]
test = os.listdir(TRAIN_DIREC)[:100]
#test = ["".join(i.split(".")[0:-1]) + "." + i.split(".")[-1] for i in test]



#resize data into img_width x img_height and prepare
def preprocess(images, path):
    x = []
    y = []
    for image in images:
        im = cv2.imread(path+image)
        im = cv2.resize(im, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
        x.append(im)
        if image[0] == 'c':
            y.append(0)
        else:
            y.append(1)
    return np.array(x), np.array(y)

x_train, y_train = preprocess(train, TRAIN_DIREC+"/")
#x_train shape (1000, 150, 150, 3) (num_training_data, img_width, img_height, channels)
#y_train shape (1000, )            (num_training_data, )
x_test, y_test = preprocess(test, TRAIN_DIREC+"/")

#build the cnn
def discriminator():
    global D
    D = Sequential()
    D.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    D.add(MaxPooling2D(pool_size=(2,2)))
    D.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    D.add(MaxPooling2D(pool_size=(2,2)))
    D.add(Flatten())
    D.add(Dense(64, activation='relu'))
    D.add(Dropout(0.5))
    D.add(Dense(1, activation='sigmoid'))


#train_datagen = ImageDataGenerator(rescale = 1./255)

#train_generator = train_datagen.flow(np.array(x_train), y_train, batch_size=batch_size) #flow generates batches of data
#D.compile(loss=keras.losses.binary_crossentropy,
                    #optimizer = keras.optimizers.RMSprop(),
                    #metrics=['accuracy'])
#D.fit_generator(train_generator, steps_per_epoch=len(x_train)//batch_size, epochs=epochs)

def generator():
    global G
    G = Sequential()
    G.add(Dense(36*36*64, input_shape=(100,)))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Reshape((36,36,64)))
    G.add(Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2)))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(32, kernel_size=(4,4), strides=(2,2)))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(3, kernel_size=(3,3), padding='same', activation='sigmoid')) #pixel values between [0.,1.]
    ##Note to plot RGB values, pixels must be either ints [0,255] or floats [0.,1.]

def stacked():
    global G
    global D
    global stacked
    stacked = Sequential()
    stacked.add(G)
    stacked.add(D) #D already has sigmoid activation

def get_batches(start, end):
    global G
    global x_train
    x_batch = x_train[start:end]
    noise_batch =  np.random.uniform(0.0, 1.0, size=(len(x_batch), 100)) #generate the same number of noisy images as in the training batch
    Gz_batch = G.predict_on_batch(noise_batch)
    return x_batch, noise_batch, Gz_batch

def train_GAN_on_batch(batches):
    global D
    global stacked
    x_batch, noise_batch, Gz_batch = batches

    #for each batch:
        #predict noise on generator: G(z) = batch of fake images
        #train real images on disciminator: D(x) = update D params per classification for real images
        #train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
        #train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

    #Update D params
    d_loss_real = D.train_on_batch(x_batch, np.ones((len(x_batch),1))) #real=1
    d_loss_fake = D.train_on_batch(Gz_batch, np.zeros((len(x_batch),1))) #fake=0
    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
    #Update G params
    g_loss = stacked.train_on_batch(noise_batch, np.ones((len(x_batch),1))) #G wants D to mark these as real
    return d_loss, g_loss #loss, accuracy tuples for each

def save_generated_images(epoch):
    rows, columns = 5, 5
    noise = np.random.uniform(0.0, 1.0, size=(rows*columns, 100))
    gen_imgs = G.predict(noise)

    fig, axs = plt.subplots(rows, columns)
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            axs[i,j].imshow(gen_imgs[cnt], interpolation='nearest')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()

def train_GAN():
    global optimizer
    global epochs
    global batch_size
    global x_train
    generator()
    discriminator()
    stacked()

    G.compile(loss=keras.losses.binary_crossentropy, optimizer = optimizer, metrics=['accuracy'])
    G.summary()
    D.compile(loss=keras.losses.binary_crossentropy, optimizer = optimizer, metrics=['accuracy'])
    D.summary()
    D.trainable = False
    stacked.compile(loss=keras.losses.binary_crossentropy, optimizer = optimizer, metrics=['accuracy'])
    stacked.summary()

    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        steps = len(x_train)//batch_size
        x_batch = None
        for i in range(steps):
            start = batch_size*i
            end = batch_size*(i+1)
            batches = get_batches(start, end)
            train_GAN_on_batch(batches)
        start = batch_size*steps
        end = len(x_train)
        batches = get_batches(start, end)
        (d_loss, d_acc), (g_loss, g_acc) = train_GAN_on_batch(batches)
        if epoch % 5 == 0:
            save_generated_images(epoch)
        print("Discriminator -- Loss:%f\tAccuracy%.2f%%\nGenerator -- Loss:%f\tAccuracy%.2f%%" %(d_loss, d_acc*100., g_loss, g_acc*100.))

train_GAN()
