from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
#from keras.applications.resnet50 import ResNet50
from PIL import Image, ImageChops
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


IMAGE_SIZE =[4892, 4892]
train_path = 'FinalDataLeft/Train'
valid_path = 'FinalDataLeft/Test'
vgg = VGG16(input_shape=IMAGE_SIZE + [3] , weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
folders = glob('FinalDataLeft/Train/*')
#folders = glob('FinalDataLeft/Test/*')

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)
# MOdel creation
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

#cost and optimization method
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('FinalDataLeft/Train',
                                                 target_size = (4892, 4892),
                                                 batch_size = 1,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('FinalDataLeft/Test',
                                            target_size = (4892, 4892),
                                            batch_size = 1,
                                            class_mode = 'categorical')

# fit the Model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')






