from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Lambda, LeakyReLU
from keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
import os, cv2, re, random
import numpy as np
import matplotlib.pyplot as plt

class DCGAN():

    def __init__(self):
        self.weights1 = []
        self.weights2 = []

        #input image dimensions
        self.img_width = 150
        self.img_height = 150
        self.input_shape = (self.img_width, self.img_height, 3) #3 channels for color

        optimizer_g = Adam(0.0002, 0.5)
        optimizer_d = Adam(0.0005, 0.5)

        self.D = self.build_discriminator()
        self.D.compile(loss=keras.losses.binary_crossentropy, optimizer = optimizer_d, metrics=['accuracy'])
        self.D.summary()

        self.G = self.build_generator()
        self.G.summary()

        '''self.D.trainable = False'''
        self.stacked = self.build_stacked()
        self.stacked.compile(loss=keras.losses.binary_crossentropy, optimizer = optimizer_g, metrics=['accuracy'])
        self.stacked.summary()

    #resize data into img_width x img_height and prepare
    def preprocess(self, images, path):
        x = []
        y = []
        for image in images:
            im = cv2.imread(path+image)
            b,g,r = cv2.split(im)       # get b,g,r
            im = cv2.merge([r,g,b])     # switch it to rgb
            im = cv2.resize(im, (self.img_width, self.img_height), interpolation = cv2.INTER_CUBIC)
            im = im * 2./255 - 1
            x.append(im)
            if image[0] == 'c':
                y.append(0)
            else:
                y.append(1)
        return np.array(x), np.array(y)


    #build the cnn
    def build_discriminator(self):
        D = Sequential()
        D.add(Conv2D(32, kernel_size=(3,3), input_shape=self.input_shape))
        D.add(LeakyReLU())
        D.add(AveragePooling2D(pool_size=(2,2)))
        D.add(Dropout(0.4))
        D.add(Conv2D(64, kernel_size=(3,3)))
        D.add(LeakyReLU())
        D.add(AveragePooling2D(pool_size=(2,2)))
        D.add(Dropout(0.4))
        D.add(Flatten())
        D.add(Dense(64))
        D.add(LeakyReLU())
        #D.add(Dropout(0.5))
        D.add(Dense(1, activation='sigmoid'))
        return D

    def build_generator(self):
        G = Sequential()
        G.add(Dense(36*36*64, input_shape=(100,)))
        G.add(BatchNormalization(momentum=0.9))
        G.add(LeakyReLU())
        G.add(Reshape((36,36,64)))
        G.add(Dropout(0.4))
        G.add(Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2)))
        G.add(BatchNormalization(momentum=0.9))
        G.add(LeakyReLU())
        G.add(Dropout(0.4))
        G.add(Conv2DTranspose(32, kernel_size=(4,4), strides=(2,2)))
        G.add(BatchNormalization(momentum=0.9))
        G.add(LeakyReLU())
        G.add(Conv2DTranspose(3, kernel_size=(3,3), padding='same', activation='tanh')) #pixel values between [0.,1.]
        ##Note: to plot RGB values with matplotlib, pixels must be either ints [0,255] or floats [0.,1.]
        return G

    def build_stacked(self):
        stacked = Sequential()
        stacked.add(self.G)
        stacked.add(self.D) #D already has sigmoid activation
        return stacked


    def get_batches(self, start, end, x_train):
        x_batch = x_train[start:end]
        noise_batch =  np.random.normal(0.0, 1.0, size=(len(x_batch), 100)) #generate the same number of noisy images as in the training batch
        Gz_batch = self.G.predict_on_batch(noise_batch)
        return x_batch, noise_batch, Gz_batch

    def train_GAN_on_batch(self, batches):
        x_batch, noise_batch, Gz_batch = batches

        #for each batch:
            #predict noise on generator: G(z) = batch of fake images
            #train real images on disciminator: D(x) = update D params per classification for real images
            #train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
            #train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        #Update D params
        self.D.trainable = True

        # self.weights2 = self.D.layers[-1].get_weights()
        # print(self.weights2[0][0][0])



        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1., size=(len(x_batch), 1)) ) #real=1, positive label smoothing
        d_loss_fake = self.D.train_on_batch(Gz_batch, np.zeros((len(x_batch), 1)) ) #fake=0
        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

        # self.weights2 = self.D.layers[-1].get_weights()
        # print(self.weights2[0][0][0])

        #Update G params
        self.D.trainable = False
        g_loss = self.stacked.train_on_batch(noise_batch, np.random.uniform(0.9, 1., size=(len(x_batch), 1)) ) #G wants D to mark these as real=1

        # self.weights2 = self.D.layers[-1].get_weights()
        # print(self.weights2[0][0][0])

        return d_loss, g_loss #loss, accuracy tuples for each

        '''x_batch, noise_batch, Gz_batch = batches
        #print('x_before: {}, y_before: {}'.format(np.shape(x_batch), np.shape(np.ones((len(x_batch),1)))))
        x = np.concatenate((x_batch, Gz_batch))
        y = np.concatenate((
                            np.ones((len(x_batch),1)),
                            np.zeros((len(x_batch),1))
                            ))
        z = list(zip(x,y))
        random.shuffle(z)
        x,y = zip(*z)
        x = np.array(list(x))
        y = np.array(list(y))
        #print('x_after: {}, y_after: {}'.format(np.shape(x), np.shape(y)))

        #for each batch:
            #predict noise on generator: G(z) = batch of fake images
            #train real images on disciminator: D(x) = update D params per classification for real images
            #train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
            #train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        # Update D params
        self.D.trainable = True
        d_loss = self.D.train_on_batch(x, y) #real=1
        #d_loss_fake = self.D.train_on_batch(Gz_batch, np.zeros((len(x_batch),1))) #fake=0
        #d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

        #print('Discriminator Loss: {}'.format(d_loss))

        # Update G params
        self.D.trainable = False
        g_loss = self.stacked.train_on_batch(noise_batch, np.ones((len(x_batch),1))) #G wants D to mark these as real
        return d_loss, g_loss #loss, accuracy tuples for each'''


    def save_generated_images(self, epoch):
        rows, columns = 5, 5
        noise = np.random.normal(0.0, 1.0, size=(rows*columns, 100))
        gen_imgs = self.G.predict(noise)

        fig, axs = plt.subplots(rows, columns)
        cnt = 0
        for i in range(rows):
            for j in range(columns):
                axs[i,j].imshow((gen_imgs[cnt] + 1)/2., interpolation='nearest')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def train_GAN(self):

        TRAIN_DIREC = 'dogs-vs-cats/train' #25,000 items
        TEST_DIREC = 'dogs-vs-cats/test1' #12,500 items

        train = os.listdir(TRAIN_DIREC)
        test = os.listdir(TEST_DIREC)

        x_train, y_train = self.preprocess(train, TRAIN_DIREC+"/")
        print(x_train.shape)
        #x_train shape (1000, 150, 150, 3) (num_training_data, img_width, img_height, channels)
        #y_train shape (1000, )            (num_training_data, )
        x_test, _ = self.preprocess(test, TEST_DIREC+"/")
        y_test = []

        epochs = 1000
        batch_size = 128

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            steps = len(x_train)//batch_size
            x_batch = None
            for i in range(steps):
                start = batch_size*i
                end = batch_size*(i+1)
                batches = self.get_batches(start, end, x_train)
                self.train_GAN_on_batch(batches)
            start = batch_size*steps
            end = len(x_train)
            batches = self.get_batches(start, end, x_train)
            (d_loss, d_acc), (g_loss, g_acc) = self.train_GAN_on_batch(batches)
            if epoch % 5 == 0:
                self.save_generated_images(epoch)
            print("Discriminator -- Loss:%f\tAccuracy%.2f%%\nGenerator -- Loss:%f\tAccuracy%.2f%%" %(d_loss, d_acc*100., g_loss, g_acc*100.))


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train_GAN()
