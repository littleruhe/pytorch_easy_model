from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, lable_dir):
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(root_dir, lable_dir)
        self.img_names = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.root_dir, self.lable_dir, img_name)
        img = Image.open(img_path)
        lable = self.lable_dir
        return img, img_name
    
    def __len__(self):
        return len(self.img_names)
    
# root_dir = "dataset/train"
# ants_lable_dir = "ants"
# bees_lable_dir = "bees"
# ants_dataset = MyData(root_dir, ants_lable_dir)
# bees_dataset = MyData(root_dir, bees_lable_dir)