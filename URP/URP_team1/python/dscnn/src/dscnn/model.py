import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.set_printoptions(threshold=sys.maxsize)


class CNNModel(nn.Module):
    def __init__(self, numF, dropoutProb, numClasses, IS_BN = True, POOLING_TYPE = "avg" ):
        super(CNNModel, self).__init__()

        numF = int(numF)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=numF,
            kernel_size=(4, 10),
            stride=(2, 1),
            padding=0,
            bias=False,
        )
        
        self.IS_BN = IS_BN
        print("IS_BN: ", IS_BN)
        
        if IS_BN:
            self.bn1 = nn.BatchNorm2d(numF)
        
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=numF,
            out_channels=numF,
            kernel_size=3,
            padding=0,
            groups=numF,
            bias=False,
        )
        
        if IS_BN:
            self.bn2 = nn.BatchNorm2d(numF)
        
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=numF, out_channels=numF, kernel_size=1, padding=0, bias=False
        )

        if IS_BN:
            self.bn3 = nn.BatchNorm2d(numF)

        self.relu3 = nn.ReLU()

        self.POOLING_TYPE = POOLING_TYPE
        if POOLING_TYPE == 'max':
            self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        elif POOLING_TYPE == 'avg':
            self.pool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        else:
            raise ValueError("Invalid pooling_type. Choose either 'max' or 'avg'.")
        

        self.dropout = nn.Dropout(dropoutProb)

        self.flatten_size = self._get_flattened_size()

        print(self.flatten_size, numClasses)
        self.fc = nn.Linear(int(self.flatten_size), int(numClasses), bias=False)
        # print(self.flatten_size)

    def _get_flattened_size(self):
        x = torch.zeros(1, 1, 10, 29)  # Create a dummy tensor
        x = self.conv1(x)
        if self.IS_BN:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        if self.IS_BN:
            x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        if self.IS_BN:
            x = self.bn3(x)
        x = self.relu3(x)

        x = self.pool4(x)
        x = x.permute(0, 2, 3, 1)

        return x.numel()

    def forward(self, x, return_outputs=False):
        outputs = {}
        # conv1 level
        x = self.conv1(x)
        if return_outputs:
            outputs["conv1"] = x.clone()
        x = self.relu1(x)

        # conv2
        x = self.conv2(x)
        if return_outputs:
            outputs["conv2"] = x.clone()
        x = self.relu2(x)

        x = self.conv3(x)
        if return_outputs:
            outputs["conv3"] = x.clone()
        x = self.relu3(x)

        x = self.pool4(x)
        if return_outputs:
            outputs["pool4"] = x.clone()

        x = self.dropout(x)
        x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        x = x.contiguous().view(x.size(0), 1, -1)

        x = self.fc(x)
        if return_outputs:
            outputs["fc"] = x.clone()

        # print(x.shape)
        if return_outputs:
            return F.log_softmax(
                x, dim=2
            ), outputs  # 평가 모드에서는 레이어별 출력도 반환
        else:
            return F.log_softmax(x, dim=2)  # 훈련 모드에서는 최종 출력만 반환


# 가중치와 바이어스를 추출하고 텍스트로 저장하는 함수
def extract_weights_and_biases(model, filename="model_parameters.txt"):
    torch.set_printoptions(precision=7)  # 소수점 이하 10자리까지 출력

    with open(filename, "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_type = "Weight" if "weight" in name else "Bias"
                f.write(f"Layer: {name}\n")
                f.write(f"Type: {layer_type}\n")
                f.write(f"Shape: {param.shape}\n")
                f.write(f"Values:\n")

                # 텐서 데이터를 그대로 저장
                f.write(f"{param.data}\n")  # 텐서 데이터를 문자열로 변환하여 그대로 저장
                f.write("\n")
