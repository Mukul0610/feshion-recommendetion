
import pickle as pkl
file_path=pkl.load(open('filepath_feshion.pkl','rb'))
feature=pkl.load(open("img_feature_feshion.pkl",'rb'))
import numpy as np
import os
#from numpy.linalg import norm
from keras.preprocessing import image
from keras.applications.resnet import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
from keras import Sequential
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import streamlit as st
st.title('Feshine Recommender system')
def feature_extraction(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expand_img=np.expand_dims(img_array,axis=0)
    final_img=preprocess_input(expand_img)
    feature= model.predict(final_img).flatten()
    final_feature=feature/np.linalg.norm(feature)
    return final_feature
model=ResNet50(include_top=False)
model.trainable=False
model=Sequential([model,GlobalMaxPooling2D()])

neighbor=NearestNeighbors(n_neighbors=5,algorithm="brute",metric="euclidean")
neighbor.fit(feature)
def save_file(uploaded_file):
    try:
        with open(os.path.join('upload_feshion',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
upload_file=st.file_uploader("Upload image")
if upload_file is not None:
    if save_file(upload_file):
        display_img=Image.open(upload_file)
        st.image(display_img)


        img_feature=feature_extraction(os.path.join("upload_feshion",upload_file.name),model)
        distance,index=neighbor.kneighbors([img_feature])
        print(index)
        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            st.image(Image.open(file_path[index[0][0]]))
        with col2:
            st.image(Image.open(file_path[index[0][1]]))
        with col3:
            st.image(Image.open(file_path[index[0][2]]))
        with col4:
            st.image(Image.open(file_path[index[0][3]]))
        with col5:
            st.image(Image.open(file_path[index[0][4]]))
