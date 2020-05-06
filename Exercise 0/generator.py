import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# In this exercise task you will implement an image generator.
# Generator objects in python are defined as having a next function.
# This next function returns the next generated object.
# In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # read labels and store label dictionary
        with open(self.label_path, 'r') as f:   # after opening of a file, a close operation should be done.
                                                # [with open() as] can be an alternative without explicitely including close()
            string = f.read()
            obj = json.loads(string)
            self.labels = obj

        # create image list based on dictionary entries
        # assumes image names starting with 0 and no empty spaces
        self.images = np.arange(0, len(self.labels), 1)

        # shuffle image list for randomness
        if (shuffle == True):
            np.random.shuffle(self.images)

        # statues variables to keep track of current index and batch size remainder
        self.index = 0

    def next(self, resize=True):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        # lists for images and labels
        batch = []
        labels = []

        # Iterate over all images in image list
        for i in range(self.index, self.index + self.batch_size):
            # restart indexing if batch size exceeds image amount
            if (i >= len(self.images)):
                i = i-len(self.images)

            # read image and label
            number = str(self.images[i])
            array = np.load(self.file_path+number+'.npy')
            name = self.labels[number]

            # resize image if flag is set
            if(resize == True):
                img = Image.fromarray(array)
                newimg = img.resize((self.image_size[0], self.image_size[1]))
                array = np.asarray(newimg)

            # augment data if mirroring or rotation flags are set
            if (self.mirroring == True or self.rotation == True):
                array = self.augment(array)

            # append images and labels
            batch.append(array)

            labels.append(name)
            # update current index
            self.index = i+1
        # return images as array -> required for automated tests
        return np.asarray(batch), labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        # mirror image if flag is set
        if (self.mirroring == True):
            if (np.random.random() >= 0.5):
                img = np.fliplr(img)

        # rotate image if flag is set
        if (self.rotation == True):
            rot = np.random.randint(1, 4)   # rotate by 90(1), 180(2) or 270(3) degrees, should start from 1 instead of 0
            img = np.rot90(img, rot)

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        name = self.class_dict.get(x)        # x is int_label not the name/number of image
        return name

    def show(self, resize=True):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next(resize)

        # arbitrary setting with 4 columns
        columns = 4
        # adjust rows according to batch size
        if(self.batch_size % columns == 0):
            rows = self.batch_size / columns
        else:
            rows = int(self.batch_size / columns) + 1

        # plot all images and labels in batch tuple
        for i in range(0, len(images)):
            plt.subplot(rows, columns, i+1)
            plt.imshow(images[i])
            plt.axis('off')
            plt.title(self.class_name(labels[i]))   # should call class_name() to obtain the titles

        plt.show()
        return
