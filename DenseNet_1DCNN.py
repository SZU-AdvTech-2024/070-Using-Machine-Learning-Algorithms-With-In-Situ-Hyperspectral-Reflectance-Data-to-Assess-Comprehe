import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, num_features, growth_rate):
        super(BottleNeck, self).__init__()
        self.num_features = num_features
        self.growth_rate = growth_rate
        self.conv1d1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.growth_rate * 4, kernel_size=1,
                                 stride=1)
        self.bn1 = nn.BatchNorm1d(self.num_features)
        self.bn2 = nn.BatchNorm1d(self.growth_rate * 4)
        self.relu = nn.ReLU()
        self.conv1d2 = nn.Conv1d(in_channels=self.growth_rate * 4, out_channels=self.growth_rate, kernel_size=3,
                                 stride=1, padding=1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        self.bn1 = self.bn1.to(x.device)
        self.bn2 = self.bn2.to(x.device)
        self.conv1d1 = self.conv1d1.to(x.device)
        self.conv1d2 = self.conv1d2.to(x.device)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1d1(x)class BottleNeck(nn.Module):
    def __init__(self, num_features, growth_rate):
        super(BottleNeck, self).__init__()
        self.num_features = num_features
        self.growth_rate = growth_rate
        self.conv1d1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.growth_rate * 4, kernel_size=1,
                                 stride=1)
        self.bn1 = nn.BatchNorm1d(self.num_features)
        self.bn2 = nn.BatchNorm1d(self.growth_rate * 4)
        self.relu = nn.ReLU()
        self.conv1d2 = nn.Conv1d(in_channels=self.growth_rate * 4, out_channels=self.growth_rate, kernel_size=3,
                                 stride=1, padding=1)
        self.drop = nn.Dropout(0.2)

        x = self.relu(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv1d2(x)
        x = self.drop(x)

        return x


class Stem(nn.Module):
    def __init__(self, num_filters):
        super(Stem, self).__init__()
        self.num_filters = num_filters
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.num_filters, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.bn = nn.BatchNorm1d(self.num_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class DenseBlock(nn.Module):
    def __init__(self, num_features, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.num_features = num_features
        self.growth_rate = growth_rate
        self.num_layers = num_layers

    def forward(self, x):
        for i in range(self.num_layers):
            bottleneck = BottleNeck(num_features=self.num_features + i * self.growth_rate, growth_rate=self.growth_rate) # !!
            out = bottleneck(x)
            x = torch.cat([x, out], dim=1)  # 拼接输入和输出
        return x


class TransitionBlock(nn.Module):
    def __init__(self, num_filters):
        super(TransitionBlock, self).__init__()
        self.num_filters = num_filters
        self.bn = nn.BatchNorm1d(self.num_filters)
        self.conv = nn.Conv1d(in_channels=self.num_filters, out_channels=self.num_filters//2, kernel_size=1, stride=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.bn = self.bn.to(x.device)
        self.conv = self.conv.to(x.device)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.pool(x)


class DenseNet121(nn.Module):
    def __init__(self, num_filters, growth_rate):
        super(DenseNet121, self).__init__()
        self.num_filters = num_filters

        self.stem = Stem(num_filters)
        self.dense_block1 = DenseBlock(num_filters, growth_rate, 6)
        num_filters_t = num_filters + 6 * growth_rate
        self.transition_block1 = TransitionBlock(num_filters_t)
        num_filters = num_filters_t // 2
        self.dense_block2 = DenseBlock(num_filters, growth_rate, 12)
        num_filters_t = num_filters + 12 * growth_rate
        self.transition_block2 = TransitionBlock(num_filters_t)
        num_filters = num_filters_t // 2
        self.dense_block3 = DenseBlock(num_filters, growth_rate, 24)
        num_filters_t = num_filters + 24 * growth_rate
        self.transition_block3 = TransitionBlock(num_filters_t)
        num_filters = num_filters_t // 2
        self.dense_block4 = DenseBlock(num_filters, growth_rate, 16)

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(320, 32)  # Adjust the input size based on the flattened dimension
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.pool2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.dense_block1(x)
        x = self.transition_block1(x)
        x = self.dense_block2(x)
        x = self.transition_block2(x)
        x = self.dense_block3(x)
        x = self.transition_block3(x)
        x = self.dense_block4(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)

        x = torch.flatten(x, 1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        # # Output layer with linear activation
        x = self.fc3(x)

        return x
