
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import torch
from torch import nn
from easy_model import EasyModel
import time


transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=transform, download=True)
valid_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transform, download=True)
train_set_sz = len(train_set)
valid_set_sz = len(valid_set)
print(f"train_set_sz={train_set_sz}, valid_set_sz={valid_set_sz}")
train_bag = DataLoader(dataset=train_set, batch_size=64, shuffle=True, drop_last=False)
valid_bag = DataLoader(dataset=valid_set, batch_size=64, shuffle=True, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择GPU还是CPU
model = EasyModel().to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
writer = SummaryWriter("./logs")

step = 0
max_acc = 0
max_epoch = 0
for epoch in range(30):
    print(f"============== epoch={epoch} ==============")

    # 训练步骤开始
    start_time = time.time()
    model.train()
    for data in train_bag:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # writer.add_images("epoch={}".format(epoch), imgs, step)
        outputs = model(imgs)
        loss = loss_func(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"step={step}, train_loss={loss.item():.2f}")
            writer.add_scalar("train_loss", loss.item(), step)
        step = step + 1
    time_use = time.time() - start_time

    # 测试步骤开始
    model.eval()
    epoch_valid_loss = 0.0
    epoch_acc = 0.0
    with torch.no_grad():
        for data in valid_bag:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_func(outputs, targets)
            epoch_valid_loss = epoch_valid_loss + loss.item()
            epoch_acc = epoch_acc + (outputs.argmax(1) == targets).sum()
    epoch_acc = epoch_acc / valid_set_sz
    print(f"epoch_train_time={time_use:.2f}s, epoch_valid_loss={epoch_valid_loss:.2f}, epoch_acc={epoch_acc:.2f}")
    writer.add_scalar("valid_loss", epoch_valid_loss, epoch)
    writer.add_scalar("valid_acc", epoch_acc, epoch)

    if epoch_acc > max_acc:
        max_acc = epoch_acc
        max_epoch = epoch
        model_path = "./model/EasyModel.pt"
        torch.save(model.state_dict(), model_path)
        # torch.save(model, "./model/EasyModel_full.pt")
print(f"min_acc={max_acc:.3f} at epoch={max_epoch}, save modle to {model_path}")


# input = torch.zeros(5, 3, 32, 32).to(device)
# torch.onnx.export(model, input, "./model/EasyModel.onnx")
# writer.add_graph(model, input)



writer.close()
