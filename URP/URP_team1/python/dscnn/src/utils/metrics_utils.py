import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 손실 그래프 저장 함수
def save_loss_plot(losses, file_path, output_file, auto =False):
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # AUTO 모드가 아닐 때만 그래프 이미지 저장
    if not auto:
        plt.savefig(file_path)  # 그래프를 이미지 파일로 저장
        plt.close()

        # 손실 값 텍스트 파일에 기록 (AUTO 모드가 아닐 때만 저장)
        with open(output_file, 'a') as f:
            f.write("Training Losses:\n")
            f.write(', '.join(map(str, losses)) + '\n')
    else:
        plt.close()  # AUTO 모드에서도 그래프 메모리 닫기

# 혼동 행렬 및 라벨 정확도 저장 함수
def save_confusion_matrix_and_accuracies(true_labels, predicted_labels, labels_list, cm_file, acc_file, auto = False):
    # 혼동 행렬 계산
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_list)

    # 각 라벨에 대한 정확도 계산
    label_accuracies = {}
    for i, label in enumerate(labels_list):
        true_positive = cm[i, i]
        total_samples = cm[i, :].sum()
        accuracy = true_positive / total_samples if total_samples > 0 else 0
        label_accuracies[label] = accuracy
    overall_accuracy = cm.diagonal().sum() / cm.sum()

    # 혼동 행렬을 텍스트 파일로 저장 (AUTO 모드가 아닐 때만 저장)
    if not auto:
        np.savetxt(cm_file, cm, fmt='%d', delimiter=',', header=','.join(map(str, labels_list)), comments='')

        # 정확도를 텍스트 파일에 저장
        with open(acc_file, 'w') as f:
            for label, accuracy in label_accuracies.items():
                f.write(f"Accuracy for label '{label}': {accuracy:.2%}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")

    # 혼동 행렬 시각화 및 저장
    if not auto:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(cm_file.replace('.txt', '.png'))  # 혼동 행렬을 이미지 파일로 저장
        plt.close()
    else:
        plt.close()  # AUTO 모드에서도 그래프 메모리 닫기
    
    return overall_accuracy

# 모델 가중치 히스토그램 저장 함수 추가
def save_weight_histograms(model, output_dir='histograms/', auto = False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 디렉토리 없으면 생성

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy()  # 텐서를 넘파이 배열로 변환
            weights_flat = weights.flatten()  # 가중치를 1D로 펼치기

            # 히스토그램 시각화 및 저장
            plt.figure(figsize=(10, 6))
            plt.hist(weights_flat, bins=50, color='blue', edgecolor='black')
            plt.title(f'Distribution of {name}')
            plt.xlabel('Weight value')
            plt.ylabel('Frequency')

            # AUTO 모드가 아닐 때만 히스토그램 저장
            if not auto:
                file_path = os.path.join(output_dir, f'{name}_histogram.png')
                plt.savefig(file_path)  # 그림을 파일로 저장
                plt.close()  # 메모리 절약을 위해 닫기
                print(f'Histogram for {name} saved at {file_path}')
            else:
                plt.close()  # AUTO 모드에서도 그래프 메모리 닫기
