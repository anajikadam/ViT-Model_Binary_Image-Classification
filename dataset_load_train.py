# from PIL import Image as img
# from IPython.display import Image, display
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
from transformers import ConvNextFeatureExtractor

image_directory_path = "Lab-Pages-data"
# Lab-Pages-data
#   -->train
#     -->class_0 : 400 .png image
#     -->class_1 : 470 .png image
#   -->test
#     -->class_0 : 9 .png image
#     -->class_1 : 9 .png image
#   -->valid
#     -->class_0 : 60 .png image
#     -->class_1 : 60 .png image

model_name = "facebook/convnext-tiny-224"
feature_extractor = ConvNextFeatureExtractor.from_pretrained(model_name)

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
transform = Compose(
    [
     RandomResizedCrop(224),
     RandomHorizontalFlip(),
     ToTensor(),
     normalize
    ]
)

def train_transforms(examples):
  examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
  return examples

def load_my_dataset():
  dataset = load_dataset("imagefolder", data_dir=image_directory_path)
  train_dataset = dataset['train']
  data = train_dataset.train_test_split(test_size=0.15)

  processed_dataset = data.with_transform(train_transforms)

  labels = data['train'].features['label'].names
  id2label = {k:v for k,v in enumerate(labels)}
  label2id = {v:k for k,v in enumerate(labels)}
  print(labels)
  print(id2label)
  print(label2id)
  return labels, id2label, label2id, processed_dataset







