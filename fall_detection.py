import time
import numpy as np
from datetime import datetime
import copy
import pymysql
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import stft
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

def compute_stft(csi_data, fs, nperseg, noverlap,noise_threshold=0.5):
    f, t, Zxx = stft(csi_data, fs, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    # Denoising by thresholding
    denoised_magnitude = np.where(magnitude > noise_threshold, magnitude, 0)

    return f, t, denoised_magnitude
def fall_detection_func(lock, detection_lock, csi_amplitude_array, csi_shape, detection_data_array, threshold1, threshold2, 
                        chosen_subcarrier=5, fs=100, stft_nperseg=60, stft_noverlap=58, high_freq_threshold=15, 
                        low_freq_threshold=10,time_window=20, duration=10,check_interval = 1):

    csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    count = 0  # Initialize count for low frequency condition duration
    last_amplitude = copy.deepcopy(csi_arr)
    check_time = time.time()  # Track the last time we checked the condition

    while True:
        current_time = time.time()
        if (last_amplitude == csi_arr).all() or current_time - check_time < check_interval:
            continue
        with lock:
            latest_data = csi_arr[:, chosen_subcarrier]

        f, t, Zxx = stft(latest_data, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
        magnitude = np.abs(Zxx)
        noise_threshold=0.4
        magnitude = np.where(magnitude > noise_threshold, magnitude, 0)

        if detection_data_array[-1] == 0:
            high_freq_avg_global = np.mean(magnitude[:time_window,high_freq_threshold:30])
            if high_freq_avg_global > threshold1.value:
                with detection_lock:
                    detection_data_array[-1] = 1
                    possible_fall_start_time = time.time()
        elif detection_data_array[-1] == 1:
            if possible_fall_start_time and (current_time - possible_fall_start_time <= duration):
                low_freq_avg_global = np.mean(magnitude[:, :low_freq_threshold])
                if low_freq_avg_global < threshold2.value:
                    count += 1
                    print(f"Low frequency condition met, count: {count}")
                    if count >= 3:  # If condition met for 5 checks, consider it a fall
                        with detection_lock:
                            detection_data_array[-1] = 2  # Update status to fall detected
                            count = 0  # Reset count after detection
                        possible_fall_start_time = None  # Reset the timer
                          
            else:
                # 如果超过了检测时间窗口，重置状态
                with detection_lock:
                    detection_data_array[-1] = 0
                    count = 0
                    possible_fall_start_time = None
                
        check_time = current_time  # Update the last check time
        time.sleep(check_interval)  # Wait for the next check interval
def fall_plot(lock, detection_data_array, cache_len, threshold1, threshold2, update_interval=100):
    # global threshold1, threshold2
    # fig,ax  = plt.subplots(2, 1, figsize=(10, 8))
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3, hspace=0.35)
    plt.title('Fall Detection Status')
    plt.xlabel('Time')
    plt.ylabel('Status')
    # 创建图形和轴
    

    # 使用轴对象的bar方法
    status_bar = ax.bar(['Fall Risk'], [0], color='green')

    # plt.show()
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['No Fall', 'Possible Fall Start', 'Fall Detected'])

    def reset_status(event):
        # 为滑动条定义回调函数
        with lock:  # 确保线程安全
            detection_data_array[-1] = 0  # 重置为“非跌倒”状态
        print("Status reset to 'No Fall'.")
        status_bar[0].set_height(0)
        status_bar[0].set_color('green')
        status_bar[0].set_label('No Fall')
        fig.canvas.draw_idle()  # 重新绘制图表以反映状态更改
    # 创建重置按钮
    reset_button_ax = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    reset_button = Button(reset_button_ax, 'Reset', color='lightgray', hovercolor='0.975')
    reset_button.on_clicked(reset_status)
    axcolor = 'lightgoldenrodyellow'

    
    # 创建滑动条
    ax_thresh1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_thresh2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    slider_thresh1 = Slider(ax_thresh1, 'Threshold 1', 0, 2, valinit=0.6)
    slider_thresh2 = Slider(ax_thresh2, 'Threshold 2', 0, 2, valinit=50)
 
    # 滑动条调整阈值的回调函数
    def update_threshold1(val):
        threshold1.value = val
    def update_threshold2(val):
        threshold2.value = val   
    slider_thresh1.on_changed(update_threshold1)
    slider_thresh2.on_changed(update_threshold2)


    # 更新状态的函数
    def update_status(frame):
        with lock:
            # 需要重新绘制图表以显示更新
            ax.figure.canvas.draw()
            detection_data = np.frombuffer(detection_data_array, dtype=np.float32).reshape((cache_len, 2))
            latest_status = detection_data[-1, 1]
            if latest_status == 0:
                status_bar[0].set_height(0)
                status_bar[0].set_color('green')
                status_bar[0].set_label('No Fall')
            elif latest_status == 1:
                status_bar[0].set_height(1)
                status_bar[0].set_color('yellow')
                status_bar[0].set_label('Possible Fall Start')
            elif latest_status == 2:
                status_bar[0].set_height(2)
                status_bar[0].set_color('red')
                status_bar[0].set_label('Fall Detected')
            ax.legend()
        return status_bar,

    ani = animation.FuncAnimation(fig, update_status, frames=np.arange(0, cache_len), interval=update_interval, blit=False)
    plt.show()

# def fall_plot(lock, detection_data_array, cache_len=100, update_interval=100):
#     global threshold1, threshold2
#     fig, ax = plt.subplots()
#     plt.title('Fall Detection Status')
#     plt.xlabel('Time')
#     plt.ylabel('Status')
#     status_bar = ax.bar(['Fall Risk'], [0], color='green')
#     ax.set_ylim(0, 2)  # Set the limit to include the "Fall Detected" status
#     ax.set_yticks([0, 1, 2])
#     ax.set_yticklabels(['No Fall', 'Possible Fall Start', 'Fall Detected'])
#     # 定义重置状态的函数
#     def reset_status(event):
#         # 为滑动条定义回调函数
#         with lock:  # 确保线程安全
#             detection_data_array[-1] = 0  # 重置为“非跌倒”状态
#         print("Status reset to 'No Fall'.")
#         status_bar[0].set_height(0)
#         status_bar[0].set_color('green')
#         status_bar[0].set_label('No Fall')
#         fig.canvas.draw_idle()  # 重新绘制图表以反映状态更改

#     # 创建重置按钮
#     reset_button_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])  # 按钮位置和大小
#     reset_button = Button(reset_button_ax, 'Reset', color='lightgray', hovercolor='0.975')
#     reset_button.on_clicked(reset_status)
#     # 添加滑动条以动态调节阈值
#     # 定义滑动条
#     def update_threshold1(val):
#         global threshold1
#         threshold1 = slider_thresh1.val

#     def update_threshold2(val):
#         global threshold2
#         threshold2 = slider_thresh2.val


#     ax_thresh1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#     ax_thresh2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#     slider_thresh1 = Slider(ax=ax_thresh1, label='Threshold 1', valmin=0, valmax=100, valinit=5)
#     slider_thresh2 = Slider(ax=ax_thresh2, label='Threshold 2', valmin=0, valmax=100, valinit=72)
#     # 注册滑动条的回调函数
#     slider_thresh1.on_changed(update_threshold1)
#     slider_thresh2.on_changed(update_threshold2)

#     def update_status(frame):
#         with lock:
#             # 正确地访问共享数组，并重塑为二维形状
#             detection_data = np.frombuffer(detection_data_array, dtype=np.float32).reshape((cache_len, 2))
#             # 更新绘图逻辑以反映二维数据的使用
#             # 示例代码省略了具体的绘图逻辑

#             latest_status = detection_data[-1, 1]  # The last element contains the latest status
#             # print(latest_status)
#             # Update the bar based on the latest detection status
#             if latest_status == 0:
#                 status_bar[0].set_height(0)
#                 status_bar[0].set_color('green')
#                 status_bar[0].set_label('No Fall')
#             elif latest_status == 1:
#                 status_bar[0].set_height(1)
#                 status_bar[0].set_color('yellow')
#                 status_bar[0].set_label('Possible Fall Start')
#             elif latest_status == 2:
#                 status_bar[0].set_height(2)
#                 status_bar[0].set_color('red')
#                 status_bar[0].set_label('Fall Detected')
                
#             ax.legend()

#         return status_bar,

#     ani = animation.FuncAnimation(fig, update_status, frames=np.arange(0, cache_len), interval=update_interval, blit=False)
#     plt.show()

