import os
from PIL import Image

shape = []
path = '/data1/haoran_yin/cq_temp/GAIA/seperate_GAIA-seg/dataset/IDD/IDD_Segmentation/leftImg8bit'
suffix = ['.jpg', '.png']
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        for suf in suffix:
            if file.endswith(suf):
                img = Image.open(file_path)
                print(img.size)
                shape.append(img.size)

print(set(shape))