
from transformers import pipeline
from transformers import ConvNextImageProcessor , AutoModelForImageClassification
# import torch

from PIL import Image as img
import os, time
# Set TF_ENABLE_ONEDNN_OPTS environment variable to 0
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st = time.time()
model_dir = "Model"
loaded_tokenizer = ConvNextImageProcessor.from_pretrained(model_dir)
loaded_model = AutoModelForImageClassification.from_pretrained(model_dir)
tm_1 = round((time.time()-st),2)
print("Model loaded...............", tm_1)

pipe = pipeline("image-classification",
                model=loaded_model,
                feature_extractor=loaded_tokenizer)

def predict(img_path):
    st = time.time()
    print(img_path)
    image = img.open(img_path)
    rs = pipe(image)
    print(rs)
    max_score = max(rs, key=lambda x: x['score'])
    max_label = max_score['label']
    max_score_value = max_score['score']

    print("prediction Label:", max_label)
    print("score:", max_score_value)
    tm_ = round((time.time()-st),2)
    print("Time: ", tm_)
    print("____"*30)
    print()

# dir_ = "Img"
# ls = os.listdir(dir_)
# for i in ls:
#     path = os.path.join(dir_, i)
#     predict(path)

