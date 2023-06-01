from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import openai, os
# load env file 
load_dotenv()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

api_key=os.getenv("OPENAI_KEY",None)

# image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
uploaded_file = st.file_uploader("Choose a Image")
if uploaded_file is not None:
    st.write(uploaded_file)
    image = Image.open(uploaded_file)
    st.image(image, caption='Input', use_column_width=True)
    text = "A photograph of"
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)

    st.write("Caption: ")
    prompt= processor.decode(out[0], skip_special_tokens=True)
    openai.api_key=api_key
    response=openai.Completion.create(
  model="text-davinci-003",
  prompt= 'You are strictly a social media caption generator that absolutely does not include any hashtags and quotation marks. Clearly label the captions "1.", "2." and "3.".\n Generate Caption for description: '+prompt,
    temperature=0.7,
    max_tokens=256
)
    

    st.write(response["choices"][0]["text"])