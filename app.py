#Works best on image size: 10-200kb
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

model = tf.keras.models.load_model('cnn_cifar10.h5')

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


uploaded_file = st.file_uploader("Upload your image", type=['jpg','png','jpeg'])
class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
                5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

if st.button("Process"):
    def preprocess_image(image):
        # Resize the image to the size expected by the model
        image = image.resize((32, 32))
        # Convert the image to a numpy array
        image_array = np.array(image)
        # Normalize the pixel values
        image_array = image_array / 255.0
        # Expand the dimensions to create a batch of size 1
        image_array = np.expand_dims(image_array, axis=0)
        return image_array



    
    if uploaded_file is not None:
        # Load the image and preprocess it
        st.write("")
        st.subheader("Processing inputs...")
        progress=st.progress(0)
        for i in range(100):
            time.sleep(0.03)
            progress.progress(i+1)
        st.balloons()
        
        image = Image.open(uploaded_file)
        cpy=image
        x = preprocess_image(image)
        # Use the model to make a prediction
        y = model.predict(x)
        # Get the predicted class label
        class_label = np.argmax(y)
        cpy = cpy.resize((120, 120))
        st.image(cpy)
        # Show the predicted class label to the user
        st.write("The predicted class label is:", class_names[class_label])
        st.subheader(class_names[class_label])





        

