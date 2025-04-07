import pathlib
import math
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS

from dscnn.dataset import get_train_and_test_set
from dscnn.data_loader import get_train_loader, get_test_loader
from dscnn.label import KEYWORDS, UNKNOWN_LABEL
from dscnn.model import CNNModel, extract_weights_and_biases
from dscnn.qat import apply_precision_change, apply_custom_quantization_to_model

from utils.metrics_utils import save_loss_plot, save_confusion_matrix_and_accuracies, save_weight_histograms
from utils.log_utils import configure_logging, log_parameter_combination
import argparse
from auto.auto import get_param_combinations  # 자동 파라미터 조합 가져오기
from auto.auto_config import *  # config의 모든 파라미터 가져오기
# import matplotlib.pyplot as plt
# import IPython.display as ipd

import sys
import os
from pathlib import Path

# dscnn 모듈이 있는 디렉토리의 절대 경로를 명시적으로 추가
# sys.path.append(str(Path("/path/to/your_project").resolve()))

from dscnn.label import KEYWORDS, UNKNOWN_LABEL

   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset predefined values
DATA_PATH = pathlib.Path("data")
NUM_NOISE_SAMPLES = 150


if DEVICE == "cuda":
    NUM_WORKERS = 1
    PIN_MEMORY = True
else:
    NUM_WORKERS = 0
    PIN_MEMORY = False

# misc parameters
LOG_INTERVAL = 20


# param 추출
def get_params_from_config(module):
    return {
        key: value
        for key, value in vars(module).items()
        if not key.startswith("__") 
        and not callable(value) 
        and not isinstance(value, type(math)) 
    }

  
import auto.auto_config as config   
params = get_params_from_config(config)

# --auto  명령어 설정
parser = argparse.ArgumentParser(description="Train and test the model.")
parser.add_argument("--auto", action="store_true", help="Run with automated parameter combinations.")
args = parser.parse_args()

# 로그 파일 설정
log_file_path = './src/result/log_folder/parameter_log.txt'
configure_logging(log_file_path)





# 데이터셋 저장(첫 한번만 실행)
import torch
import pickle

# # 데이터셋 준비 함수
# def prepare_datasets(data_path, num_noise_samples):
#     train_set, test_set = get_train_and_test_set(data_path, num_noise_samples)
#     return train_set, test_set

# 데이터셋과 로더 설정 저장 함수
# def save_datasets_and_loaders(train_set, test_set, loader_settings, train_file='train_set.pkl', test_file='test_set.pkl', loader_file='loader_settings.pkl'):
#     # 데이터셋 저장
#     with open(train_file, 'wb') as f:
#         pickle.dump(train_set, f)
#     with open(test_file, 'wb') as f:
#         pickle.dump(test_set, f)
    
#     # 로더 설정 저장
#     with open(loader_file, 'wb') as f:
#         pickle.dump(loader_settings, f)
# def load_params_from_file(filename):
#     with open(filename, 'r') as f:
#         for line in f:
#             key, value = line.strip().split('=')
#             if key in globals():
#                 globals()[key] = float(value) if value.replace('.', '', 1).isdigit() else value

# # 미리 데이터셋을 준비하고 저장
# train_set, test_set = prepare_datasets(DATA_PATH, NUM_NOISE_SAMPLES)
# loader_settings = {
#     'batch_size': BATCH_SIZE,
#     'num_workers': NUM_WORKERS,
#     'pin_memory': PIN_MEMORY
# }

# save_datasets_and_loaders(train_set, test_set, loader_settings)

# # 데이터셋과 로더 설정 로드 함수
def load_datasets_and_loaders(train_file='train_set.pkl', test_file='test_set.pkl', loader_file='loader_settings.pkl'):
    # 데이터셋 로드
    with open(train_file, 'rb') as f:
        train_set = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_set = pickle.load(f)
    
    # 로더 설정 로드
    with open(loader_file, 'rb') as f:
        loader_settings = pickle.load(f)

    return train_set, test_set, loader_settings



def create_loaders(train_set, test_set, loader_settings):
    train_loader = get_train_loader(train_set, **loader_settings)
    test_loader = get_test_loader(test_set, **loader_settings)
    return train_loader, test_loader


# 저장된 데이터셋과 로더 설정을 불러와서 데이터 로더 재생성
train_set, test_set, loader_settings = load_datasets_and_loaders()

# 데이터 로더 다시 생성
train_loader, test_loader = create_loaders(train_set, test_set, loader_settings)
####


# CNN 모델 정의
def num_of_correct(pred, target):
    """
    count number of correct predictions
    """
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    """
    find most likely label index for each element in the batch
    """
    return tensor.argmax(dim=-1)


def train(model, epoch,optimizer):
    model.train()

    for idx, (data, _, target) in enumerate(train_loader):
        data = data.to(device=DEVICE)
        target = target.to(device=DEVICE)
        
        data = apply_precision_change(data, int(IN_INT), int(IN_FRAC))

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)

        loss.backward()
        
        for param in model.parameters():
            param.data = apply_precision_change(param.data, int(W_INT), int(W_FRAC))
            
        optimizer.step()
        apply_custom_quantization_to_model(model, int(W_INT), int(W_FRAC))


def test(model, epoch):
    model.eval()
    correct = 0
    total_n = len(test_loader.dataset)

    for data, _, target in test_loader:
        data = data.to(device=DEVICE)
        target = target.to(device=DEVICE)

        output = model(data)
        pred = get_likely_index(output)
        correct += num_of_correct(pred, target)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct / total_n} ({100. * correct / total_n:.0f}%)\n"
    )
    
def predict(tensor,model):

    tensor = tensor.to(device=DEVICE)

    tensor = tensor.unsqueeze(0)  # (1, 1, height, width)로 확장


    tensor = model(tensor)
    tensor = get_likely_index(tensor)
    tensor = tensor.squeeze()
    return tensor

# 모델 한번 돌릴때 실행되는 코드
def run_experiment(params,test_loader):
    model = CNNModel(
        numF=params["NUM_F"],
        dropoutProb=params["DROPOUT_PROB"],
        numClasses=params["NUM_CLASSES"],
        IS_BN=params["IS_BN"],
        POOLING_TYPE=params["POOLING_TYPE"]
    ).to(device=DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["STEP_SIZE"], gamma=params["GAMMA"])

    EPOCHS = int(params["EPOCHS"])
    with tqdm(total=EPOCHS) as pbar:
        for epoch in range(1, EPOCHS + 1):
            train(model, epoch,optimizer)
            accuracy = test(model, epoch)
            scheduler.step()
            pbar.update(1)



    # ===== 예측 및 혼동 행렬 생성 =====
    subset = list(test_loader.dataset)[1928:2330]  # 테스트 셋 일부 사용
    true_labels = []
    predicted_labels = []

    for waveform, _, utterance, *_ in subset:
        output = predict(waveform, model)
        # 텐서일 경우 CPU로 이동 후 int로 변환
        output = output.cpu().item() if isinstance(output, torch.Tensor) else output
        utterance = utterance.cpu().item() if isinstance(utterance, torch.Tensor) else utterance

        true_labels.append(utterance)
        predicted_labels.append(output)

    # 혼동 행렬 생성 및 정확도 저장
    labels_list = list(range(12))  # 라벨 리스트 (0~11)
    overall_accuracy = save_confusion_matrix_and_accuracies(
        true_labels, predicted_labels, labels_list,
        './src/result/confusion_matrix.txt',
        './src/result/results.txt',
        args.auto
    )

    return overall_accuracy, model


# auto 모드(반복 시행)
if args.auto:
    print("\nauto\n")
    param_combinations = get_param_combinations(params)
    for i, param_set in enumerate(param_combinations, start=1):
        # print(f"Running Experiment {i} with params: {param_set}")
        overal_accuracy,_ = run_experiment(param_set,test_loader)

        extra_parameters = {"KEYWORDS": ', '.join(KEYWORDS), "Overall Accuracy": f"{overal_accuracy:.2%}"}
        log_parameter_combination(param_set, extra_parameters=extra_parameters)
# 1번 시행
else:
    accuracy, model = run_experiment(params,test_loader)

    extra_parameters = {"Overall Accuracy": f"{accuracy:.2%}"}
    log_parameter_combination(params, extra_parameters=extra_parameters)

    extract_weights_and_biases(model, './src/result/weight_Q/parameter_Q.txt')
    save_weight_histograms(model, './src/result/weight_Q', args.auto)