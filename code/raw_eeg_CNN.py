# coding:UTF-8
'''
对原始的 EEG 信号，使用 CNN 进行情感分类。
Created by Xiao Guowen.
'''
from utils.tools import build_preprocessed_eeg_dataset_CNN, RawEEGDataset, subject_independent_data_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchviz
import numpy as np
from io import BytesIO
import torchvision
from PIL import Image
import io
import torchvision.transforms as transforms


# 加载数据，整理成所需要的格式
folder_path = "D:/数据集/SEED/Preprocessed_EEG"
feature_vector_dict, label_dict = build_preprocessed_eeg_dataset_CNN(folder_path)
train_feature, train_label, test_feature, test_label = subject_independent_data_split(feature_vector_dict, label_dict,
                                                                                      {'2', '6', '9'})

desire_shape = [1, 62, 200]
train_data = RawEEGDataset(train_feature[:int(len(train_feature) * 0.01)], 
                           train_label[:int(len(train_label) * 0.01)], 
                           desire_shape)
test_data = RawEEGDataset(test_feature[:int(len(test_feature) * 0.01)], 
                          test_label[:int(len(test_label) * 0.01)], 
                          desire_shape)

# 超参数设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 30
num_classes = 3
batch_size = 24
learning_rate = 0.0001

# Data loader
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)


# 定义卷积网络结构
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128 * 7 * 25, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)


# Train the model
def train(writer):
    total_step = len(train_data_loader)
    batch_cnt = 0
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_data_loader):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            batch_cnt += 1
            writer.add_scalar('train_loss', loss, batch_cnt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
        scheduler.step()

        # 在每个epoch结束后进行测试，传递当前的epoch
        test(writer, epoch)  # 传递 epoch 参数给 test()

    torch.save(model.state_dict(), './code/model/model.ckpt')


# Test the model
def test(writer, epoch, is_load=False):
    if is_load:
        model.load_state_dict(torch.load('./code/model/model.ckpt'))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_data_loader:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Test Accuracy after Epoch {epoch} is {100 * correct / total:.2f}%')



def plot_confusion_matrix(cm, epoch, writer):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f'./code/logs/confusion_matrix_epoch_{epoch}.png')
    plt.close()

    # Log confusion matrix image to TensorBoard
    # 在 plot_confusion_matrix 中调用
    writer.add_image('Confusion Matrix', plt_to_image(plt), epoch)



def plt_to_image(plt):
    # 将当前的 plt 图像保存到内存中的 BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # 返回文件开头

    # 使用 PIL 打开图像
    image = Image.open(buf)

    # 转换为 Tensor
    transform = transforms.ToTensor()
    return transform(image)


def plot_gradient_distribution(model, epoch, writer):
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.flatten().cpu().detach().numpy())
    all_grads = np.concatenate(all_grads)

    plt.figure(figsize=(6, 5))
    plt.hist(all_grads, bins=50, color='blue', alpha=0.7)
    plt.title(f'Gradient Distribution - Epoch {epoch}')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.savefig(f'./code/logs/gradient_distribution_epoch_{epoch}.png')
    plt.close()

    # Log gradient distribution image to TensorBoard
    writer.add_image('Gradient Distribution', plt_to_image(plt), epoch)


# 绘制模型结构图
def plot_model_structure(model, sample_input):
    y = model(sample_input)
    dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_structure", format="png")
    print("Model structure saved as model_structure.png")


# 在训练开始时，绘制模型结构图
sample_input = torch.randn(1, 1, 62, 200).to(device)
# plot_model_structure(model, sample_input)

# 初始化TensorBoard的writer
writer = SummaryWriter('../log')

if __name__ == '__main__':
    writer = SummaryWriter('../log')
    train(writer)  # 确保传递 writer
    test(writer, epoch=0, is_load=True)  # 调用 test() 时传递 epoch 参数
