import torch
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
#
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# model = AutoModel.from_pretrained('facebook/dinov2-base')
#
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs[0]

