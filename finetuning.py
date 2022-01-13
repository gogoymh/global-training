import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

from dataset import cifar_part
from preact_resnet_simclr import resnet56, Output_layer

train_loader = DataLoader(cifar_part(), batch_size=128, shuffle=True)#, pin_memory=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        './data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ),
                        ),
                batch_size=64, shuffle=False)#, pin_memory=True)


device = torch.device("cuda:0")
model = resnet56().to(device)
final = Output_layer().to(device)


path = "/data/ymh/global_training/global_trained4.pth"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])


params = list(model.parameters()) + list(final.parameters())
optimizer = optim.Adam(params, lr=0.01)
#optimizer = optim.Adam(final.parameters(), lr=0.01)

#optimizer = optim.SGD(params, lr=0.05, momentum=0.9, nesterov=True)
#optimizer = optim.SGD(final.parameters(), lr=0.05, momentum=0.9, nesterov=True)

#optimizer = optim.SGD(model.parameters(), lr=0.1)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(90):
    model.train()
    final.train()
    runnning_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        
        #with torch.no_grad():
        #    mid = model(x.float().to(device))
        
        mid = model(x.float().to(device))
        output = final(mid)
        loss = criterion(output, y.long().to(device))
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
        
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        final.eval()
        correct = 0
        for x, y in test_loader:
            with torch.no_grad():
                mid = model(x.float().to(device))
                output = final(mid)
            
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)

        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % accuracy)
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % accuracy)

        
    #scheduler.step()
    
    #torch.save({'model_state_dict': model.state_dict()}, "/data/ymh/gpu_test/resnet56_preact.pth")