import fitz
import numpy as np
# import pandas as pd
import os, time
# from pathlib import Path
from PIL import Image

from transformers import pipeline
from transformers import ConvNextImageProcessor , AutoModelForImageClassification
# import torch

print("Model loading")
st = time.time()
model_dir = "Model"
loaded_tokenizer = ConvNextImageProcessor.from_pretrained(model_dir)
loaded_model = AutoModelForImageClassification.from_pretrained(model_dir)
pipe = pipeline("image-classification",
                model=loaded_model,
                feature_extractor=loaded_tokenizer)

tm_1 = round((time.time()-st),2)
print("Model loaded...............", tm_1)

def predict_page(img):
    st = time.time()
    rs = pipe(img)
    max_score = max(rs, key=lambda x: x['score'])
    # max_score['score'] = round(max_score['score'], 2)
    # max_label = max_score['label']
    # max_score_value = max_score['score']
    tm_ = round((time.time()-st),5)
    return max_score

def write_txt_file(sent = [], dir_ = '/', input_file_name = "output"):
    if isinstance(sent, (list, tuple)):
        sent_ = '\n'.join(sent)
    elif isinstance(sent, str):
        sent_ = sent
        
    fileName = input_file_name+'.txt'
    filePath = os.path.join(dir_, fileName)
    with open(filePath, mode='w', encoding='utf-8') as fp:
        fp.write(sent_)
    print("--- Text file Saved........")


# read saved output txt file
def read_txt_file(dir_, file_name = "output"):
    fileName = file_name+'.txt'
    filePath = os.path.join(dir_, fileName)
    with open(filePath, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
    return lines

def get_req_pages(lines):
    for i in lines:
        if 'Class_1:' in i:
            class_1 = i.split(':')[1].strip()
            #print(class_1)
    pgs = [int(i)-1 for i in class_1.split(',')]
    print("number of Class 1 pages: ", len(pgs))
    return pgs


def pdf_page_predict_filePath(pdf_file_path):
    dir_ = pdf_file_path.replace(os.path.basename(pdf_file_path), '')
    
    doc = fitz.open(pdf_file_path)
    num_pages = len(list(doc.pages()))
    print("PDF file Path: ",pdf_file_path)
    print("Number of Pages: ", num_pages)

    i = 0
    res_data = []
    for page in doc:
        i += 1
        pix = page.get_pixmap(dpi=100)
        buf_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        arr_img = np.ascontiguousarray(buf_img)
        img = Image.fromarray(arr_img)
        result = predict_page(img)
        result['PageID'] = i
        #print(result)
        res_data.append(result)
        # break
        
    filtered_class_0 = [i for i in res_data if i['label']=='class_0']
    filtered_class_1 = [i for i in res_data if i['label']=='class_1']
    Class_1 = [i['PageID'] for i in filtered_class_1]
    Class_0 = [i['PageID'] for i in filtered_class_0]
    dir1      = "FolderDir:  {:<10}".format(dir_)
    filepath  = "FilePath:  {:<10}".format(pdf_file_path)
    cs_0 = "Class_0:  {:<10}".format(",".join([str(i) for i in Class_0]))
    cs_1 = "Class_1:  {:<10}".format(",".join([str(i) for i in Class_1]))

    ls1 = ["| PageID {:<5}| label {:<5}| score {:<22}|"\
        .format(i['PageID'], i['label'], i['score']) for i in res_data]
    ls1.append(dir1)
    ls1.append(filepath)
    ls1.append(cs_0)
    ls1.append(cs_1)
    write_txt_file(ls1, dir_=dir_ )
    print("Done")

# try:
#     lines = read_txt_file(dir_=dir_)
#     req_pages_indx = get_req_pages(lines)
# except FileNotFoundError:
#     # Handle the FileNotFoundError exception
#     print("File not found. Please make sure the file exists.")
#     req_pages_indx = list(range(num_pages))

# ftz_pgs = list(doc.pages())
# # ftz_pgs = ftz_pgs[21:24]
# ftz_pgs = [ftz_pgs[i] for i in req_pages_indx]
# print("After filter number of pages: ", len(ftz_pgs))


# dir_ = "Downloads/17027_10514"
def pdf_page_predict(dir_):
    pdf_file_path = [os.path.join(dir_,i) for i in os.listdir(dir_) if i.endswith('.pdf')][0]
    doc = fitz.open(pdf_file_path)
    num_pages = len(list(doc.pages()))
    print("PDF file Path: ",pdf_file_path)
    print("Number of Pages: ", num_pages)

    i = 0
    res_data = []
    for page in doc:
        i += 1
        pix = page.get_pixmap(dpi=100)
        buf_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        arr_img = np.ascontiguousarray(buf_img)
        img = Image.fromarray(arr_img)
        result = predict_page(img)
        result['PageID'] = i
        print(result)
        res_data.append(result)
        # break
        
    filtered_class_0 = [i for i in res_data if i['label']=='class_0']
    filtered_class_1 = [i for i in res_data if i['label']=='class_1']
    Class_1 = [i['PageID'] for i in filtered_class_1]
    Class_0 = [i['PageID'] for i in filtered_class_0]
    filepath  = "FilePath:  {:<5}".format(pdf_file_path)
    cs_0 = "Class_0:  {:<5}".format(",".join([str(i) for i in Class_0]))
    cs_1 = "Class_1:  {:<5}".format(",".join([str(i) for i in Class_1]))

    ls1 = ["| PageID {:<5}| label {:<5}| score {:<22}|"\
        .format(i['PageID'], i['label'], i['score']) for i in res_data]
    ls1.append(filepath)
    ls1.append(cs_0)
    ls1.append(cs_1)
    write_txt_file(ls1, dir_=dir_ )

    try:
        lines = read_txt_file(dir_=dir_)
        req_pages_indx = get_req_pages(lines)
    except FileNotFoundError:
        # Handle the FileNotFoundError exception
        print("File not found. Please make sure the file exists.")
        req_pages_indx = list(range(num_pages))

    ftz_pgs = list(doc.pages())
    # ftz_pgs = ftz_pgs[21:24]
    ftz_pgs = [ftz_pgs[i] for i in req_pages_indx]
    print("After filter number of pages: ", len(ftz_pgs))


dir_ = "Downloads/17027_10514"
print(dir_)
pdf_page_predict(dir_)
print("End Program..")


