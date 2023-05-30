from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
import numpy as np
from PIL import Image

# Defing Model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Upload image 
uploaded_file = st.file_uploader("Choose a file")

# check if image is uploaded
if uploaded_file is not None:
    st.write(uploaded_file)
    image = Image.open(uploaded_file)
    st.image(image, caption='Input', use_column_width=True)
    text = "A photograph of"
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)
    
    st.write("Caption: ")
    st.write(processor.decode(out[0], skip_special_tokens=True))