# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os


#testdata_image_path = './data/test/'#png图片的地址

class MyDataset(Dataset):
 def __init__(self, txt_path, train=True, transform = None, target_transform = None):
     
    self.data_path = txt_path
    self.train_flag = train
    
    filenames = [name for name in os.listdir(txt_path)
                     if os.path.splitext(name)[-1] == '.png'] #选择指定目录下的.png图片
    imgs = []
    for i, filename in enumerate(filenames):
        img_adrr = os.path.join(txt_path, filename)
    
        imgs.append((img_adrr, filename))
        self.imgs = imgs 

        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.Resize(size = (32,32)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform     


 def __getitem__(self,  index: int):
    fn, label = self.imgs[index]
    img = Image.open(fn).convert('RGB') 
    if self.transform is not None:
        img = self.transform(img) 
    return img, label

 def __len__(self):
    return len(self.imgs)
