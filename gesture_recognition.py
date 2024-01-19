import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class resnet(nn.Module):
    def __init__(self,channel=1,class_num=8):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, class_num)

    def forward(self,x):
        # batch, channel, length, carrier = x.shape

        # x = x.permute(0, 1, 3, 2)
        # x = x.reshape(-1, carrier, length)
        # x = torch.bmm(x, x.permute(0, 2, 1))
        # x = x.view(batch, channel, carrier, carrier)

        # x=x.view(-1, length, carrier)
        # x=torch.bmm(x,x.permute(0,2,1))
        # x=x.view(batch,channel,length,length)


        x=self.model(x)
        return x


def gesture_recognition(lock, csi_amplitude_array,csi_shape,result_lock,action_array,people_array,model_path="./model",task_action=6,task_people=8):
    model_action=resnet(class_num=task_action)
    model_action.load_state_dict(torch.load(model_path+"/action.pth",map_location=device))
    model_action=model_action.to(device)
    model_action.eval()
    model_people=resnet(class_num=task_people)
    model_people.load_state_dict(torch.load(model_path+"/people.pth",map_location=device))
    model_people=model_people.to(device)
    model_people.eval()
    torch.set_grad_enabled(False)

    csi_arr=np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    action_arr=np.frombuffer(action_array, dtype=np.float32).reshape(task_action)
    people_arr=np.frombuffer(people_array, dtype=np.float32).reshape(task_people)


    while True:
        with lock:
            data=torch.from_numpy(csi_arr)
        data=data.reshape([1,1,csi_shape[0],csi_shape[1]]).to(device)
        # data=data[:,:,:,:52]
        data = data / torch.norm(data, dim=-2, keepdim=True)
        action=model_action(data)
        people=model_people(data)
        action=torch.softmax(action,dim=-1)
        people=torch.softmax(people,dim=-1)
        with result_lock:
            action_arr[:]=action.detach().cpu().squeeze().numpy()
            people_arr[:]=people.detach().cpu().squeeze().numpy()

action_name=["applause","waveright","upanddown","frontandback","circleclockwise","leftandright"]
action_name_short=["app","wav","updown","frontback","circle","leftright"]
people_name=['czj','ctw','lxh','chl','wyt','mfy','zzj','jyh']

def gesture_recognition_plot(action_array,people_array,task_action=6,task_people=8):
    action_arr = np.frombuffer(action_array, dtype=np.float32).reshape(task_action)
    people_arr = np.frombuffer(people_array, dtype=np.float32).reshape(task_people)

    plt.ion()  # 打开交互模式
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13, 5))
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax1.set_ylabel("Action")
    # ax1.set_xticklabels(action_name, rotation='vertical', fontsize=7, fontweight='bold')  # 将横坐标标签纵向显示
    ax2.set_ylabel("People")
    ax2.yaxis.set_label_position('right')
    bar_action = ax1.bar(action_name_short, action_arr)  # 创建柱状图
    bar_people = ax2.bar(people_name, people_arr)  # 创建柱状图

    while True:
        for bar, h in zip(bar_action, action_arr):
            bar.set_height(h)  # 更新柱状图的高度
        ax1.set_title("Action: "+action_name[np.argmax(action_arr)])

        for bar, h in zip(bar_people, people_arr):
            bar.set_height(h)  # 更新柱状图的高度
        ax2.set_title("People: "+people_name[np.argmax(people_arr)])

        plt.draw()
        plt.pause(0.3)  # 暂停一段时间，使得图形有动态效果
