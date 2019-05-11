# dog_cat_gan
DCGAN to generate images of dogs and cats from the Kaggle Dogs vs. Cats dataset at https://www.kaggle.com/c/dogs-vs-cats/data

Currently under construction; hyperparameter tuning is underway.

![Checkerboard pattern at epoch 15](https://raw.githubusercontent.com/niharikajainn/dog_cat_gan/master/images/15.png)

Nearing the beginning of training (epoch 15), the generator learns no meaningful features existing in the data and generates images with the same checkerboard pattern.

![Checkerboard pattern at epoch 195](https://raw.githubusercontent.com/niharikajainn/dog_cat_gan/master/images/195.png)

After about 200 epochs, the generator has learned a reasonable style of coloring and places an animal-like object in the center of images. The learning did not find much further success; this might be due to too much variance in the original data. These images were all of different (and high) dimensions and had varying positions of the animal in the photo. This discriminator architecture also did not distinguish between dogs and cats. Future work might use smaller, more homogeneous images (such as only using images cropped to the animal face) and use a conditional GAN to further classify "real" images as depicting a dog or cat.

