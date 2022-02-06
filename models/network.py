import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Net(nn.Module):
    def __init__(self, dataset_type):
        super(Net, self).__init__()

        if dataset_type == 'sumnist':
            self.conv1 = nn.Conv2d(2, 32, 3, 1)
        elif dataset_type == 'diffsumnist':
            self.conv1 = nn.Conv2d(3, 32, 3, 1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x.squeeze(1)


class Resnet(nn.Module):
    def __init__(self, dataset_type):
        super(Resnet, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=False, num_classes=1)

        if dataset_type == 'sumnist':
            self.model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif dataset_type == 'diffsumnist':
            self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.model(x)
        return x.squeeze(1)
