# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:44:46 2019

@author: Cool
"""

import os, signal
import tensorflow as tf

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np


#from google.colab import files
from keras.preprocessing import image

#import zipfile

#local_zip = 'cats_and_dogs_filtered.zip'

#zip_ref = zipfile.ZipFile(local_zip, 'r')

#zip_ref.extractall()
#zip_ref.close()

#%%

base_dir = 'C:/Users/cool/Documents/Analytics/tensorFlow/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')

# Directory with training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#%%
""" let's see what the filenames look like in the cats and dogs train directories (file naming conventions are the
 same in the validation directory """
 
train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

#%%

""" 
find out the total number of cat and dog images in the train and validation directories:
    """
    
print('total training cat images :', len(os.listdir(      train_cats_dir ) ))
print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))

print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))

#%%
"""
let's take a look at a few pictures to get a better sense of what the cat and dog datasets look like.
configure the matplot parameters also:
"""



# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images

#%%
""" display a batch of 8 cat and 8 dog pictures. You can rerun the cell to see a fresh batch each time: """
# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cat_fnames[ pic_index-8:pic_index] 
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

#%%
"""there are now 4 convolutional layers with 32, 64, 128 and 128 convolutions respectively. """

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
""" 

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
    
model.summary()
    
#%%
from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.optimizers import Adagrad


model.compile(optimizer=RMSprop(lr=1e-4),
#model.compile(optimizer = Adagrad(lr = 1e-4),
              loss='binary_crossentropy',
              metrics = ['acc'])

#%%
"""
Let's set up data generators that will read pictures in our source folders, convert them to float32 tensors,
 and feed them (with their labels) to our network. We'll have one generator for the training images and one
 for the validation images. Our generators will yield batches of 20 images of size 150x150 and their 
 labels (binary).

Data that goes into neural networks should usually be normalized in some way to make
 it more amenable to processing by the network. (It is uncommon to feed raw pixels into a convnet.) In our 
 case, we will preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally
 all values are in the [0, 255] range).

In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale
 parameter. This ImageDataGenerator class allows you to instantiate generators of augmented image batches
 (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then
 be used with the Keras model methods that accept data generators as inputs: fit_generator,
 evaluate_generator, and predict_generator.

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
#train_datagen = ImageDataGenerator( rescale = 1.0/255. )
# This code has changed. Now instead of the ImageGenerator just rescaling
# the image, we also rotate and do other operations
# Updated to do image augmentation to avoid overfitting, where training accuracy is vry good but test is not so good
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))     
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

#%%
"""
Let's train on all 2,000 images available, for 15 epochs, and validate on all 1,000 test images. (This may
 take a few minutes to run.)

Do note the values per epoch. 
You'll see 4 values per epoch -- Loss, Accuracy, Validation Loss and Validation Accuracy.

The Loss and Accuracy are a great indication of progress of training. It's making a guess as to the
 classification of the training data, and then measuring it against the known label, calculating the
 result. Accuracy is the portion of correct guesses. The Validation accuracy is the measurement with the
 data that has not been used in training. As expected this would be a bit lower. You'll learn about why
 this occurs in the section on overfitting later.
"""

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_steps=50,
                              verbose=2)

#%%
"""
Let's now take a look at actually running a prediction using the model. This code will allow you to choose
 1 or more files from your file system, it will then upload them, and run them through the model, giving an
 indication of whether the object is a dog or a cat.
"""
#Doesnt work..needs changes

#uploaded=files.upload()
val_dir = 'C:/Users/cool/Documents/Analytics/tensorFlow/testdata'

dirlist = os.listdir(val_dir)

#for fn in uploaded.keys():
for fn in dirlist: 
  # predicting images
  #path = '/content/' + fn
  path = val_dir + '/' + fn
  img = image.load_img(path, target_size=(150, 150))
  
  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=10)
  
  print(classes[0])
  
  if classes[0]>0:
    print(fn + " is a dog")
    
  else:
    print(fn + " is a cat")

#%%
"""plot the training/validation accuracy and loss as collected during training: """

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )   

#%% 
"""
Another way to create the above plot

"""
#import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%%
"""  terminate the kernel and free memory resources: """
os.kill(     os.getpid(), signal.SIGTERM )

#%%
