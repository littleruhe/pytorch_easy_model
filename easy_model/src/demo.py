import torch
from easy_model import EasyModel
from self_data import MyData
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def show_imgs(imgs):
    for img, name in imgs:
        print(name)

dataset = MyData("./dataset/test", "")
show_imgs(dataset)
transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor()])
imgs = torch.stack([transform(item[0]) for item in dataset])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择GPU还是CPU

model = EasyModel().to(device)
model.load_state_dict(torch.load("./model/EasyModel.pt", weights_only=True))
model.eval()

writer = SummaryWriter("./logs")
imgs = imgs.to(device)
writer.add_images("imgs", imgs, 1)
with torch.no_grad():
    outputs = model(imgs)
    
print(outputs)
print(outputs.argmax(1)) #'airplane=0, automobile=1, bird=2, cat=3, der=4, dog=5, frog=6, horse=7, ship=8, trunk=9

writer.close()