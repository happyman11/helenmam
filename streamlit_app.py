#%%
###import packages


import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import time
import random
import numpy as np

from keras.models import load_model
def load_model_trained(img):
    img_array = np.array(img)
    img_array=img_array.resize(1,64,64,3)
    model_loaded = load_model('./WEIGHTS/cnn_lstm.h5')
    st.write(model_loaded.summary())
    prediction=model_loaded.predict(img_array)
    
    return(prediction)

#functions

def resize_image(img_array):
      img_array=img_array.resize((64, 64), Image.ANTIALIAS)
      img_array = np.array(img_array)
      img_array=img_array.resize(1,64,64,4)
      return(img_array)
    
def read_image(path):

    image = Image.open(path)
    return(image)
##


def read_uploaded_image(path):

    image = Image.open(path)
    img_64=image.resize((64, 64), Image.ANTIALIAS)
    img_array = np.array(image)
    return(image,img_array,img_64)

st.title("***Radar Based Activity Recognition using CNN-LSTM***")      
#sidebar

#file upload testing
st.sidebar.header("Activity Prediction")



st.sidebar.subheader("Upload Image")
image_predict = st.sidebar.file_uploader("Upload your input file", type=["png","jpg","jpeg"])


#dataset display

st.subheader('Dataset: INSHEP')

with st.spinner('Loading Dataset..'):
     time.sleep(2)


col1, col2, col3 = st.beta_columns(3)
with col1:
    st.write("***Dataset type***: Classification")
with col2:
    st.write("***Activity***: 6 activities")
with col3:
    st.write("***Shape***: 1D complex array")


col1, col2, col3 = st.beta_columns(3)

with col1:
   path_label1="./Images/Label1surfacespectrogram.jpg"
   label1_img=read_image(path_label1)
   st.image(label1_img, caption='Label: Walking Back',use_column_width=True)
   
with col2:
   path_label2="./Images/label2surface.jpg"
   label2_img=read_image(path_label2)
   st.image(label2_img, caption='Label: Walking Forth',use_column_width=True)

with col3:
   path_label3="./Images/Label3surface.jpg"
   label3_img=read_image(path_label3)
   st.image(label3_img, caption='Label: Sitting of chair',use_column_width=True)


col4, col5, col6 = st.beta_columns(3)


with col4:
   path_label4="./Images/Label4surface.jpg"
   label4_img=read_image(path_label4)
   st.image(label4_img, caption='Label: Standing up',use_column_width=True)
   
with col5:
   path_label5="./Images/Label5surface.jpg"
   label5_img=read_image(path_label5)
   st.image(label5_img, caption='Label: Picking up Object',use_column_width=True)

with col6:
   path_label6="./Images/Label6Surface.jpg"
   label6_img=read_image(path_label3)
   st.image(label6_img, caption='Label: Picking up glass or cup',use_column_width=True)


#dataset Preprocessing

with st.spinner('Loading Preprocessed Dataset....'):
     time.sleep(2)

st.subheader('Dataset: Preprocessing')


#Trainning



st.subheader('Model: Trainning and Testing')

with st.spinner('Loading Information..'):
     time.sleep(2)


if(st.sidebar.button("Predict")):
    
    col7,col8=st.beta_columns(2)
    with st.spinner('Processing Input..'):
        time.sleep(5)
        
    with col7:
        Prediction_image,image_array,img_64=read_uploaded_image(image_predict)
        
        st.image(Prediction_image, caption='Uploaded Image',use_column_width=True)

    with col8:
       prediction=load_model_trained(img_64)
       with st.spinner('Predicting.....'):
           time.sleep(4)
       st.write(prediction)
       
             

    














    
































col10, col11= st.beta_columns(2)
with col10:







  
    components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    

 <p> <span style="color:red;font-weight: bold"> Phd Project:</span>&nbsp A. Helen Victoria,<br>
  <span> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp Assistant Professor,</span><br>
  <span> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp School of Computing,</span><br>
  <span> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp<a href="https://www.srmist.edu.in/it-dept/faculty/a-helen-victoria">Profile</a></span></p>
 

"""
)

with col11:
  
    components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    

 <p> <span style="color:red;font-weight: bold"> Affiliation:</span>&nbsp SRMIST, Kattankulathur,<br>
 <span> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp  Assistant Professor,</span><br>
 <span> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp Chennai, India</span></p>

"""
)



    
    
