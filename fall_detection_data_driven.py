import torchvision.models as models
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F

# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 25 * 13, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def fall_recognition(lock, csi_amplitude_array,csi_shape,result_lock,action_array,model_path="F:\SRIBD\ESP32-Realtime-System\Falldataset_20240225\dataset_20240225.pth",task_action=3):
    model_action=SimpleCNN(num_classes=task_action)
    model_action.load_state_dict(torch.load(model_path,map_location=device))
    model_action=model_action.to(device)
    model_action.eval()
    torch.set_grad_enabled(False)

    csi_arr=np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    action_arr=np.frombuffer(action_array, dtype=np.float32).reshape(task_action)

    while True:
        with lock:
            data=torch.from_numpy(csi_arr)
        data=data.reshape([1,1,csi_shape[0],csi_shape[1]]).to(device)
        # data=data[:,:,:,:52]
        # data = data / torch.norm(data, dim=-2, keepdim=True)
        action=model_action(data)
        action=torch.softmax(action,dim=-1)
        with result_lock:
            action_arr[:]=action.detach().cpu().squeeze().numpy()

action_name=["fall","sit","walk"]

def fall_recognition_plot(action_array,task_action=3):
    action_arr = np.frombuffer(action_array, dtype=np.float32).reshape(task_action)

    plt.ion()  # 打开交互模式
    fig, ax1 = plt.subplots()
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Action")
    # ax1.set_xticklabels(action_name, rotation='vertical', fontsize=7, fontweight='bold')  # 将横坐标标签纵向显示
    bar_action = ax1.bar(action_name, action_arr)  # 创建柱状图

    while True:
        for bar, h in zip(bar_action, action_arr):
            bar.set_height(h)  # 更新柱状图的高度
        ax1.set_title("Action: "+action_name[np.argmax(action_arr)])

        plt.draw()
        plt.pause(0.3)  # 暂停一段时间，使得图形有动态效果
