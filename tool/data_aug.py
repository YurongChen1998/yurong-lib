import os
from PIL import Image, ImageFilter 

path = './contamination/'
data_aug_path = './contamination_aug'

os.mkdir(data_aug_path)

def resize(img, factor):
    # The input img should be Image, factor should be int

    h, w = img.size
    return img.resize((int(h*factor), int(w*factor)))

for image in os.listdir(path):
    ori_img = Image.open(os.path.join(path, image))
    ori_img.save(os.path.join(data_aug_path, str('ori_' + image)), 'png')

    '''
    Resize image 
    '''
    tmp_img = resize(ori_img, 0.5)
    tmp_img.save(os.path.join(data_aug_path, str('down2_' + image)), 'png')
    tmp_img = resize(ori_img, 0.25)
    tmp_img.save(os.path.join(data_aug_path, str('down4_' + image)), 'png')
    tmp_img = resize(ori_img, 0.125)
    tmp_img.save(os.path.join(data_aug_path, str('down8_' + image)), 'png')
    tmp_img = resize(ori_img, 2)
    tmp_img.save(os.path.join(data_aug_path, str('up2_' + image)), 'png')
    tmp_img = resize(ori_img, 4)
    tmp_img.save(os.path.join(data_aug_path, str('up4_' + image)), 'png')
    tmp_img = resize(ori_img, 8)
    tmp_img.save(os.path.join(data_aug_path, str('up8_' + image)), 'png')
    # ___________________________________________________ # 

    '''
    Noise image 
    '''
    tmp_img = ori_img.filter(ImageFilter.GaussianBlur(radius = 5))
    tmp_img.save(os.path.join(data_aug_path, str('G5_' + image)), 'png')
    tmp_img = ori_img.filter(ImageFilter.GaussianBlur(radius = 3))
    tmp_img.save(os.path.join(data_aug_path, str('G3_' + image)), 'png')
    tmp_img = ori_img.filter(ImageFilter.GaussianBlur(radius = 2))
    tmp_img.save(os.path.join(data_aug_path, str('G2_' + image)), 'png')
    # ___________________________________________________ # 

    '''
    Rotate image 
    '''
    tmp_img = ori_img.rotate(270)
    tmp_img.save(os.path.join(data_aug_path, str('rot270_' + image)), 'png')
    tmp_img = ori_img.rotate(90)
    tmp_img.save(os.path.join(data_aug_path, str('rot90_' + image)), 'png')
    #tmp_img = ori_img.rotate(135)
    #tmp_img.save(os.path.join(data_aug_path, str('rot135_' + image)), 'png')
    tmp_img = ori_img.rotate(180)
    tmp_img.save(os.path.join(data_aug_path, str('rot180_' + image)), 'png')
    # ___________________________________________________ # 
