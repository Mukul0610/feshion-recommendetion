import os
import pickle as pkl
img_paths=os.listdir("image_feshion")
import numpy as np
#from numpy.linalg import norm
from keras.preprocessing import image
from keras.applications.resnet import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
from keras import Sequential
from tqdm import tqdm

model=ResNet50(include_top=False)
model.trainable=False

model=Sequential([model,GlobalMaxPooling2D()])

def feature_extraction(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expand_img=np.expand_dims(img_array,axis=0)
    final_img=preprocess_input(expand_img)
    feature= model.predict(final_img).flatten()
    final_feature=feature/np.linalg.norm(feature)

    return final_feature

img_feature=[]
for i in tqdm(img_paths):
    img_feature.append(feature_extraction((os.path.join("image_feshion", i)), model))
pkl.dump(img_feature,open("img_feature_feshion.pkl","wb"))








