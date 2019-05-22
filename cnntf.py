# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:10:30 2019

@author: Cool
"""

import os, signal
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

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

import tensorflow as tf

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

model.compile(optimizer=RMSprop(lr=0.001),
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
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
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
                              epochs=15,
                              validation_steps=50,
                              verbose=2)

#%%
"""
Let's now take a look at actually running a prediction using the model. This code will allow you to choose
 1 or more files from your file system, it will then upload them, and run them through the model, giving an
 indication of whether the object is a dog or a cat.
"""
#Doesnt work..needs changes

import numpy as np


#from google.colab import files
from keras.preprocessing import image

#uploaded=files.upload()
val_dir = 'C:/Users/cool/Documents/Analytics/tensorFlow/cats_and_dogs_filtered/validation/testdata'

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
"""
To get a feel for what kind of features our convnet has learned, one fun thing to do is to visualize how an
 input gets transformed as it goes through the convnet.

Let's pick a random cat or dog image from the training set, and then generate a figure where each row is the
output of a layer, and each image in the row is a specific filter in that output feature map. Rerun this
cell to generate intermediate representations for a variety of training images.

"""

import numpy as np
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------

    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
    
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
As you can see, we are overfitting like it's getting out of fashion. Our training accuracy (in blue) gets
 close to 100% (!) while our validation accuracy (in green) stalls as 70%. Our validation loss reaches its
 minimum after only five epochs.

Since we have a relatively small number of training examples (2000), overfitting should be our number one
 concern. Overfitting happens when a model exposed to too few examples learns patterns that do not 
 generalize to new data, i.e. when the model starts using irrelevant features for making predictions. 
 For instance, if you, as a human, only see three images of people who are lumberjacks, and three images 
 of people who are sailors, and among them the only person wearing a cap is a lumberjack, you might start
 thinking that wearing a cap is a sign of being a lumberjack as opposed to a sailor. You would then make a
 pretty lousy lumberjack/sailor classifier.

Overfitting is the central problem in machine learning: given that we are fitting the parameters of our
 model to a given dataset, how can we make sure that the representations learned by the model will be 
 applicable to data never seen before? How do we avoid learning things that are specific to the training
 data?

In the next exercise, we'll look at ways to prevent overfitting in the cat vs. dog classification model.

"""
#%%
"""  terminate the kernel and free memory resources: """
os.kill(     os.getpid(), signal.SIGTERM )

#%%
