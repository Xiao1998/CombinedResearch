import torch
import torchvision
from torchvision import transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import torchvision.transforms as T

class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, padding = 1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel, 3, stride, padding, bias=False),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, padding, bias=False),
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ReResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, padding = 1, shortcut=None):
        super(ReResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.ConvTranspose2d(inchannel, outchannel, 3, stride, padding, bias=False),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.ConvTranspose2d(outchannel, outchannel, 3, 1, padding, bias=False)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.pre = nn.Conv2d(3, 16, 3,padding=1, bias=False)

        self.layer1 = self._make_layer(16, 16*6, 3,stride=1,padding=1,size_change=True,short_size=1,short_padding=0,short_stride=1)
        #self.subsampleing2 =nn.Conv2d(16, 16,3, stride=2,padding=1)
        self.layer2 = self._make_layer(16*6, 32*6, 3,stride=2,padding=1,size_change=True,short_stride=2,short_padding=1,short_size=3)
        #self.subsampleing3 =nn.Conv2d(32, 32,3, stride=2,padding=1)
        self.layer3 = self._make_layer(32*6, 64*10, 3,stride=2,padding=1,size_change=True,short_stride=2,short_padding=1,short_size=3)
        self.norm_layer = nn.Sequential(
            nn.BatchNorm2d(64*10),
            nn.ReLU()
        )
        #self.fc = nn.Linear(512*12*12, 256*2*2)
        self.fc2 = nn.Linear(64*10, num_classes)
        self.reform = nn.Sequential(
            nn.ConvTranspose2d(640,640,5),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.ConvTranspose2d(640,640,4),
            nn.BatchNorm2d(640),
            nn.ReLU()
        )
        self.relayer1 = self.re_make_layer(640, 320, 3,stride=2,padding=1,size_change=True,short_stride=2,short_padding=1,short_size=3)
        self.relayer2 = self.re_make_layer(320, 160, 3,stride=2,padding=1,size_change=True,short_stride=2,short_padding=1,short_size=3)
        self.formsize = nn.ConvTranspose2d(160,160,4)
        self.relayer3 = self.re_make_layer(160, 16, 3,stride=1,padding=1,size_change=True,short_size=3,short_padding=1,short_stride=1)
        self.finaltrim = nn.ConvTranspose2d(16,3,3,padding=1)



    def _make_layer(self, inchannel, outchannel, block_num, stride=1,padding =1,short_size = 3,short_stride= 1,short_padding = 1,size_change = True):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, short_size, stride=short_stride,padding=short_padding, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        if size_change == False:
            layers.append(ResidualBlock(inchannel, outchannel, padding=padding, stride=stride, shortcut=None))
        else:
            layers.append(ResidualBlock(inchannel, outchannel, padding=padding, stride=stride, shortcut=shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def re_make_layer(self, inchannel, outchannel, block_num, stride=1,padding =1,short_size = 3,short_stride= 1,short_padding = 1,size_change = True):
        shortcut = nn.Sequential(
            nn.ConvTranspose2d(inchannel, outchannel, short_size, stride=short_stride,padding=short_padding, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        if size_change == False:
            layers.append(ReResidualBlock(inchannel, outchannel, padding=padding, stride=stride, shortcut=None))
        else:
            layers.append(ReResidualBlock(inchannel, outchannel, padding=padding, stride=stride, shortcut=shortcut))
        for i in range(1, block_num):
            layers.append(ReResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)


    def encode(self,x):
        x = self.pre(x)
        # print(x.size())
        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        x = self.norm_layer(x)
        # print(x.size())
        x = F.avg_pool2d(x, 8,1,0)
        # print(x.size())
       # x = x.view(x.size(0), -1)
        return x

    def decode(self,x):
        x = self.reform(x)
        #print(x.size())
        x = self.relayer1(x)
        #print(x.size())
        x = self.relayer2(x)
        x = self.formsize(x)
        #print(x.size())
        x = self.relayer3(x)
        #print(x.size())
        x = self.finaltrim(x)
        #print(x.size())
        return x

    def final_train(self,x):
        x = x.view(-1, 3, 32, 32)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        y = encoded.view(encoded.size(0), -1)
        y = self.fc2(y)
        return y,encoded,decoded

    def forward(self, x):
       x = x.view(-1,3,32,32)
       x = self.pre(x)
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.norm_layer(x)
       x = F.avg_pool2d(x,8,1,0)
       x = x.view(x.size(0),-1)
       return self.fc2(x)


if __name__ == '__main__':
    net = ResNet()
    ini_epoch = 0

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    transform = T.Compose([
        T.Pad(4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.RandomCrop(32),
        transform
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    criterion = nn.CrossEntropyLoss()
    criterion_auto = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    def load_model(file,optimizer,net):
        dict_state = torch.load(file)
        net.load_state_dict(dict_state["model"])
        optimizer.load_state_dict(dict_state["opt"])
        net.train()
        ini_epoch = dict_state["epoch"]
        return net,optimizer,ini_epoch

    #net,optimizer,ini_epoch = load_model("model_widen.pkl",optimizer,net)


    for epoch in range(ini_epoch,120):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
           # inputs = inputs.cuda()
           # labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch % 50 ==0 and i == 0:
                lr = 0.1 * (0.2** math.floor(epoch / 50))
                for param in optimizer.param_groups:
                    param['lr'] = lr
                print("Change rage: " + str(lr))
            running_loss += loss.data[0]
            if i % 2000 == 1999:
                torch.save(dict(model =net.state_dict(), opt =optimizer.state_dict(),epoch =epoch),"model_widen.pkl")
                print('[%d, %.5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("Done!")
    net.eval()

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = images.cuda()
    labels = labels.cuda()

   # plt.imshow(torchvision.utils.make_grid(images[0].permute(2,1,0)))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
       # print(Variable(images))
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test image: %d %%' %(
        100 * correct / total
    ))

    class_correct = list(0 for i in range(10))
    class_total = list(0 for i in range(10))
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data,1)
        c  = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    print(classes)
    print(class_correct)
    print(class_total)

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]
        ))