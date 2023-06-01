---
title: Demo App
emoji: ⚡
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.21.0
app_file: app.py
pinned: false
---
# Image Captioning Model - BLIP

## Introduction
 Image Captioning Model - BLIP (Bootstrapping Language-Image Pre-training). This model is designed for unified vision-language understanding and generation tasks. It is trained on the COCO (Common Objects in Context) dataset using a base architecture with a ViT (Vision Transformer) large backbone.
 
 It utilizes the BLIP architecture, which combines bootstrapping language-image pre-training with the ability to generate creative captions using the OpenAI ChatGPT API.

The image captioning model is implemented using the PyTorch framework and leverages the Hugging Face Transformers library for efficient natural language processing.

Used streamlit python library for creating interactive web applications.

## Demo
Caption can be generate for any image at [link](https://huggingface.co/spaces/luv-bansal/demo-app)


## Example Images with Generated Captions
Here are some example images along with the captions generated by the BLIP image captioning model:

![Image 1](images/football.jpeg)

**Generated Caption:** "Nothing beats the joy of a sunny day spent playing soccer with friends."

![Image 2](images/jeep-woods.jpg)

**Generated Caption:** 
 * Nature is calling, so answer the call with your Jeep and let the adventure begin.
 * Live life on the wild side and take the road less traveled.


![Image 3](images/sunset.jpeg)

**Generated Caption:**
 * Take a moment to appreciate the beauty of a sunset by the beach.
 * The beach is the perfect place to end the day and enjoy the beauty of the sunset.



## Requirements
To run the image captioning model, the following dependencies are required:
- Python (version 3.7 or above)
- PyTorch (version 1.8 or above)
- Transformers library (version 4.3 or above)

You can install the necessary libraries using the following command:

```
pip install -r requirements.txt
```

## Acknowledgments
We would like to express our gratitude to the researchers and developers who have contributed to the development and implementation of the BLIP image captioning model. Their dedication and hard work are greatly appreciated.

## Contact
If you have any questions, issues, or feedback regarding the image captioning model, please feel free to contact us at [bansal22luvi@gmail.com](mailto:bansal22luvi@gmail.com).

We hope you find the BLIP image captioning model useful and enjoy experimenting with it!

