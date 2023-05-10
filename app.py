import pandas as pd
import streamlit as st
import numpy as np
import time

import streamlit as st

st.title('Image Classifier!')
st.title(f'Our Project is _trained using_ :blue[CNN model] and utilizes the CIFAR-10 dataset :sunglasses:')
st.write("")
st.write("")
st.write("A project by: Kavya Sharma and Rudra Danak.")

st.header("About the dataset:-")
st.subheader('The CIFAR-10 dataset \n')
st.write("The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.\nThe dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.")
st.image('cifar_kr.png')

code='''
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
'''

st.write("")
st.write("")
st.write("CNN model:")
st.code(code, language="python")
st.metric("Accuracy", "CNN", "72%")

st.write("")
st.write("")

def loader():
        st.write("")
        st.subheader("Processing inputs...")
        progress=st.progress(0)
        for i in range(100):
            time.sleep(0.03)
            progress.progress(i+1)
        st.balloons()
        st.warning("Feature will be added soon!")

st.file_uploader("Upload image to classfiy...")
st.button("Process",key='bt1',on_click=loader)

