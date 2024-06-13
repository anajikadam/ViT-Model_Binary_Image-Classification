import os
from program import predict

dir_ = "Img"
ls = os.listdir(dir_)
for i in ls:
    path = os.path.join(dir_, i)
    predict(path)

