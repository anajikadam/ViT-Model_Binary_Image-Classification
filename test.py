import fitz
import numpy as np
# import pandas as pd
import os, time
from pathlib import Path
from PIL import Image
# from imp_page_predict import write_txt_file

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

def create_input_file():
    ls = []
    root_dir = "Downloads"
    list_dirs = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]
    for idx, dir_ in enumerate(list_dirs):
        print(idx+1, dir_)
        try:
            pdf_file_path = [os.path.join(dir_,i) for i in os.listdir(dir_) if i.endswith('.pdf')][0]
            #output_file_path = [os.path.join(dir_,i) for i in os.listdir(dir_) if i.endswith('output.txt')][0]
            lines = read_txt_file(dir_=dir_)
            req_pages_indx = get_req_pages(lines)
        except FileNotFoundError:
            # Handle the FileNotFoundError exception
            print("File not found. Please make sure the file exists.")
            # req_pages_indx = list(range(num_pages))
            req_pages_indx = []
        finally:
            # ftz_pgs = [ftz_pgs[i] for i in req_pages_indx]
            # print("After filter number of pages: ", len(ftz_pgs))
            sent = "FilePath: {};ImpPages:{}".format(pdf_file_path, ",".join([str(i) for i in req_pages_indx]))
            print(sent)
            ls.append(sent)
        print("____"*30)
    write_txt_file(sent = ls, dir_ = 'filter_page_input/', input_file_name = "imp_input")

lines = read_txt_file(dir_ = 'filter_page_input/', file_name = "imp_input")
# print(lines[0])
ln = lines[0]

def get_class_1_pages(ln):
    FilePath = ln.split(';')[0].replace('FilePath:', '').strip()
    ImpPages = ln.split(';')[1].replace('ImpPages:', '').strip()
    req_pages_indx  = [int(i) for i in ImpPages.split(',')]
    return FilePath, req_pages_indx

rs = get_class_1_pages(ln)
print(rs)

