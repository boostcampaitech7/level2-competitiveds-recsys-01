import torch
from torch import nn
import torch.optim as optim

# CNN 모듈 (위/경도 처리)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 7, 128)  # Input을 적절히 Flatten하여 Linear Layer에 전달
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutional layers
        x = self.pool(self.relu(self.conv1(x)))  # (N, 1, 43, 28) -> (N, 32, 21, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (N, 32, 21, 14) -> (N, 64, 10, 7)

        # Flatten the output of the conv layers
        x = x.view(-1, 64 * 10 * 7)  # Flatten: (N, 64, 10, 7) -> (N, 64 * 10 * 7)

        # Fully connected layer
        x = self.relu(self.fc1(x))  # (N, 64 * 10 * 7) -> (N, 128)
        return x

# MLP 모듈 (정형 데이터 처리: 금리, 면적 등)
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 96)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.squeeze(1)
        return x

# 결합 모델 (CNN + MLP)
class CombinedModel(nn.Module):
    def __init__(self, input_size):
        super(CombinedModel, self).__init__()
        self.cnn = CNNModel()
        self.mlp = MLPModel(input_size)
        self.fc = nn.Linear(128+96, 1)  # CNN과 MLP의 출력 크기를 합친 후 최종 예측

    def forward(self, x_cnn, x_mlp):
        cnn_out = self.cnn(x_cnn)  # CNN 처리 (위/경도)
        mlp_out = self.mlp(x_mlp)  # MLP 처리 (정형 데이터)
        combined = torch.cat((cnn_out, mlp_out), dim=1)  # 두 출력을 결합
        out = self.fc(combined)  # 최종 예측 (전세가)
        return out