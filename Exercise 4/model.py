from data import ChallengeDataset
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('data.csv')
train_data, val_data = train_test_split(data, test_size=0.2)
train_set = ChallengeDataset(train_data, 'train')
val_set = ChallengeDataset(val_data, 'val')
train_dl = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
val_dl = DataLoader(val_set, batch_size=1, num_workers=2)



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                                      nn.BatchNorm2d(num_features=64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.resBlock1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(num_features=64),
                                       nn.ReLU(inplace=True))
        self.resBlock1_residual = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
                                                nn.BatchNorm2d(num_features=64))
        self.resBlock2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(num_features=128),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(num_features=128),
                                       nn.ReLU(inplace=True))
        self.resBlock2_residual = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2),  # stride = 2 !!!!
                                                nn.BatchNorm2d(num_features=128))
        self.resBlock3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(num_features=256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(num_features=256),
                                       nn.ReLU(inplace=True))
        self.resBlock3_residual = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2),
                                                nn.BatchNorm2d(num_features=256))

        self.resBlock4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(num_features=512),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(num_features=512),
                                       nn.ReLU(inplace=True))
        self.resBlock4_residual = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2),
                                                nn.BatchNorm2d(num_features=512))
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=2),
                                        nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        res1 = self.resBlock1_residual(x.clone())
        x = self.resBlock1(x)
        x = res1 + x

        res2 = self.resBlock2_residual(x.clone())
        x = self.resBlock2(x)
        x = res2 + x

        res3 = self.resBlock3_residual(x.clone())
        x = self.resBlock3(x)
        x = res3 + x

        res4 = self.resBlock4_residual(x.clone())
        x = self.resBlock4(x)
        x = res4 + x

        x = self.globalAvgPool(x)

        out = self.classifier(x.flatten(start_dim=1))

        return out






