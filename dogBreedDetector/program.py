from keras import activations, optimizers
from keras.preprocessing.image import load_img
from keras.saving.save import load_model
import numpy as np
from numpy.core.numeric import base_repr
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import os,sys

from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping

base_dir='.'
data_dir=os.path.join(base_dir,'train')
files=os.listdir(data_dir)

labels=pd.read_csv(os.path.join(base_dir,'labels.csv'))
#print(labels.head())

file_df=pd.DataFrame({'id':list(map(lambda x:x.replace('.jpg',''),files))})
#print(file_df.head())

#mapping files with breed
label_info=pd.merge(left=file_df,right=labels)
#print(label_info.head())

#converting id to onehot encoded notation
num_classes=len(label_info.breed.unique())
#print(num_classes)

le=LabelEncoder()
breed=le.fit_transform(label_info.breed)
Y=np_utils.to_categorical(breed,num_classes=num_classes)

#print(Y.shape)

#converting images to numpy array
input_dim=(50,50)
X=np.zeros((Y.shape[0],*input_dim,3))

for i,img in enumerate(files):
    image=load_img(os.path.join(data_dir,img),target_size=input_dim)
    image=img_to_array(image)
    image=image.reshape((1,*image.shape))
    image=preprocess_input(image)
    X[i]=image
#print(X.shape)

earlystop=EarlyStopping(monitor='val_loss', min_delta=0,patience=2,verbose=0,mode='auto')

from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Flatten,Dropout

vgg_model=VGG19(weights='imagenet',include_top=False)
x=vgg_model.output
x=GlobalAveragePooling2D()(x)
x=Dropout(0.2)(x)
out=Dense(120,activation='softmax')(x)

model=Model(inputs=vgg_model.input,outputs=out)

for layer in vgg_model.layers:
    layer.trainable=False
    
from tensorflow.keras.optimizers import Adam
opt=Adam()
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
#print(model.summary())

history_last_layer=model.fit(X,Y,batch_size=128,epochs=10,validation_split=0.2,verbose=2,callbacks=[earlystop],)
model.save('modellastlayer.h5')

imgVerf=load_img('pug.jpg',target_size=input_dim)
imgVerf=img_to_array(imgVerf)
imgVerf=imgVerf.reshape((1,*imgVerf.shape))
imgVerf=preprocess_input(imgVerf)


last_layer_model=load_model('modellastlayer.h5')
res=last_layer_model.predict(imgVerf)
s=np.argsort(res)[0][-5:]
print(le.inverse_transform(s))
