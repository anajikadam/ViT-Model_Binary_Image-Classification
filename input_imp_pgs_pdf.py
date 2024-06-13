import os
    # read saved output txt file
def read_txt_file(dir_, file_name = "output"):
    fileName = file_name+'.txt'
    filePath = os.path.join(dir_, fileName)
    with open(filePath, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
    return lines

lines = read_txt_file(dir_ = 'filter_page_input/', file_name = "imp_input")
# print(lines[0])
ln = lines[0]

def get_class_1_pages(ln):
    FilePath = ln.split(';')[0].replace('FilePath:', '').strip()
    ImpPages = ln.split(';')[1].replace('ImpPages:', '').strip()
    req_pages_indx  = [int(i) for i in ImpPages.split(',')]
    return FilePath, req_pages_indx


for i in range(len(lines)):
    rs = get_class_1_pages(lines[i])
    print(rs)

