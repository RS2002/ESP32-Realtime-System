import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
from scipy.signal import stft
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def show_csi_func(lock,csi_amplitude_array,cache_len,csi_shape):
    fig, ax = plt.subplots()
    plt.title('csi-amplitude')
    plt.xlabel('packets')
    plt.ylabel('amplitude')
    ax.set_ylim(0, 40)
    ax.set_xlim(0, cache_len)
    x = np.arange(0, cache_len, 1)
    csi_arr=np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)

    # 选择载波10、20、30进行显示
    line1, = ax.plot(x, np.abs(csi_arr[: ,10]), linewidth=1.0, label='subcarrier_10')
    line2, = ax.plot(x, np.abs(csi_arr[: ,20]), linewidth=1.0, label='subcarrier_20')
    line3, = ax.plot(x, np.abs(csi_arr[: ,30]), linewidth=1.0, label='subcarrier_30')
    plt.legend()

    def init():
        line1.set_ydata([np.nan] * len(x))
        line2.set_ydata([np.nan] * len(x))
        line3.set_ydata([np.nan] * len(x))
        return line1, line2, line3,

    def animate(i):
        with lock:
            line1.set_ydata(np.abs(csi_arr[: ,10]))
            line2.set_ydata(np.abs(csi_arr[: ,20]))
            line3.set_ydata(np.abs(csi_arr[: ,30]))
        return line1, line2, line3,

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000 / 25, blit=True,cache_frame_data=False)
    plt.show()


def show_csi_heatmap_func(lock, csi_amplitude_array, cache_len, csi_shape):
    fig, ax = plt.subplots()
    plt.title('CSI Amplitude Heatmap')
    plt.xlabel('Packet Index')
    plt.ylabel('Subcarrier Index')
    # 假设csi_shape为(cache_len, 52)，52个载波
    csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    # 创建初始热力图，这里使用np.abs(csi_arr)的第一个时间点的数据，实际上应该根据需要显示的数据进行调整
    csi_heatmap = ax.imshow(np.abs(csi_arr).T, aspect='auto', origin='lower', 
                            extent=[0, cache_len, 0, csi_shape[1]], 
                            interpolation='none', cmap='viridis')
    fig.colorbar(csi_heatmap, ax=ax, orientation='vertical', label='Amplitude')

    def animate(i):
        with lock:
            # 更新热力图的数据。这里简单地重新计算了csi_arr，但实际上你可能需要根据新数据更新它
            csi_heatmap.set_data(np.abs(csi_arr).T)
        return csi_heatmap,

    ani = animation.FuncAnimation(fig, animate, interval=1000 / 25, blit=False)
    plt.show()




def show_csi_complex_func(lock, csi_amplitude_array, csi_phase_array, cache_len, csi_shape):
    # 初始化复数CSI矩阵
    csi_amplitude_matrix = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    csi_phase_matrix = np.frombuffer(csi_phase_array, dtype=np.float32).reshape(csi_shape)
    complex_csi_matrix = csi_amplitude_matrix * np.exp(1j * csi_phase_matrix)

    fig, ax = plt.subplots()
    plt.title('Current CSI Complex Plane Visualization')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    
    # 为了只显示当前的CSI包，我们初始化一个空的散点图
    scatter = ax.scatter([], [], s=10)
    
    def animate(i):
        with lock:
            # 更新复数CSI数据
            csi_amplitude_matrix = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
            csi_phase_matrix = np.frombuffer(csi_phase_array, dtype=np.float32).reshape(csi_shape)
            complex_csi_matrix = csi_amplitude_matrix * np.exp(1j * csi_phase_matrix)
#             # 归一化复数CSI矩阵
            max_val = np.max(np.abs(complex_csi_matrix))
            normalized_csi_matrix = complex_csi_matrix / max_val
            # 选择一个示例的当前时间点，这里简单选取最后一个时间点的数据
            current_data = normalized_csi_matrix[-1, :]  # 假设最后一个时间点为当前数据
            
            # 更新散点图的数据为当前时间点的CSI数据
            scatter.set_offsets(np.column_stack((current_data.real, current_data.imag)))

            # 归一化处理可以根据需要添加
            
        return scatter,

    # 设置动画更新和初始化函数
    ani = animation.FuncAnimation(fig, animate, interval=100, blit=False)

    # 设置图表的显示范围
    plt.xlim(-1, 1)  # 根据实际的数据范围进行调整
    plt.ylim(-1, 1)  # 根据实际的数据范围进行调整
    plt.grid(True)
    plt.show()


# def show_csi_STFT_func(lock, csi_amplitude_array, cache_len, csi_shape, fs=8000):
#     """
#     Display STFT heatmap of a single subcarrier's CSI data.
#     """
#     # Reshape and prepare CSI data
#     csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
#     # Select a single subcarrier's data
#     single_subcarrier_data = csi_arr[:, 0]
    
#     # Compute STFT for the selected subcarrier
#     f, t, Zxx = stft(single_subcarrier_data, fs, nperseg=256, noverlap=128)
#     magnitude = np.abs(Zxx)

#     # Plot setup
#     fig, ax = plt.subplots()
#     plt.title('STFT Magnitude Heatmap of Subcarrier 0')
#     plt.xlabel('Time [sec]')
#     plt.ylabel('Frequency [Hz]')
    
#     # Initial heatmap
#     stft_heatmap = ax.imshow(magnitude, aspect='auto', origin='lower', 
#                              extent=[t.min(), t.max(), f.min(), f.max()], 
#                              interpolation='none', cmap='viridis')
#     fig.colorbar(stft_heatmap, ax=ax, orientation='vertical', label='Magnitude')
    
#     plt.show()
    
# def show_csi_STFT_func(lock, csi_amplitude_array, cache_len, csi_shape, fs=100, update_interval=20):
#     """
#     Display STFT heatmap of CSI data considering cache_len for latest data.
    
#     Parameters:
#     - lock: A threading or multiprocessing lock to ensure data consistency.
#     - csi_amplitude_array: Shared memory array containing CSI amplitude data.
#     - cache_len: Length of the data cache to consider for STFT.
#     - csi_shape: Shape of the CSI data (time, subcarriers).
#     - fs: Sampling frequency of the CSI data.
#     """
#     # Ensure the lock is acquired to maintain data consistency
#     with lock:
#         # Reshape CSI data
#         csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
        
#     # Initialize plot
#     fig, ax = plt.subplots()
#     plt.title('Dynamic STFT Magnitude Heatmap')
#     plt.xlabel('Time [sec]')
#     plt.ylabel('Frequency [Hz]')
    
#     # Placeholder for the heatmap
#     stft_heatmap = ax.imshow(np.zeros((csi_shape[1]//2+1, cache_len)), aspect='auto', origin='lower', 
#                              interpolation='none', cmap='viridis')
#     fig.colorbar(stft_heatmap, ax=ax, orientation='vertical', label='Magnitude')

#     def update_heatmap(frame):
#         with lock:
#             # Determine the starting point to consider cache_len of latest data
#             # start_point = max(0, csi_arr.shape[0] - cache_len)
#             # latest_data = csi_arr[start_point:, 5]  # Assuming analysis on the first subcarrier
#             # 计算每个子载波时间序列的平均值，并从时间序列中减去
#             # csi_arr_dc_removed = csi_arr - np.mean(csi_arr, axis=0)

#             # 接下来你可以对去除直流分量后的数据执行STFT
#             # 选择某个子载波的数据进行STFT
#             subcarrier_index = 10  # 示例子载波索引
#             # latest_data = csi_arr[:, subcarrier_index]

#             # Compute STFT on the latest data
#             f, t, Zxx = stft(latest_data, fs, nperseg=30, noverlap=28)
#             magnitude = np.abs(Zxx)
#             # noise_threshold = 0.4
#             # magnitude = np.where(magnitude > noise_threshold, magnitude, 0)
#             # Update heatmap data
#             stft_heatmap.set_data(magnitude)
#             stft_heatmap.set_extent([t.min(), t.max(), f.min(), f.max()])
#             stft_heatmap.set_clim(vmin=0, vmax=5)
        
#         return stft_heatmap,
    
#     # Create an animation that updates the heatmap
#     ani = animation.FuncAnimation(fig, update_heatmap, interval=update_interval, blit=False, cache_frame_data=False)
    
#     plt.show()

def show_csi_STFT_func(lock, csi_amplitude_array, cache_len, csi_shape, fs=100, update_interval=100):
    with lock:
        # Reshape CSI data
        csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
        
    # 初始化图表和轴
    fig, ax1 = plt.subplots()#2, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.3, hspace=0.35)

    # STFT热图轴
    stft_heatmap = ax1.imshow(np.zeros((csi_shape[1]//2 + 1, 100)), aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(stft_heatmap, ax=ax1, orientation='vertical', label='Magnitude')
    ax1.set_title('STFT Magnitude Heatmap')
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Frequency [Hz]')


    def update(frame):
        with lock:

            # 假设csi_amplitude_array是全局变量或以其他方式访问
            # 计算STFT
            # csi_arr_dc_removed = csi_arr - np.mean(csi_arr, axis=0)

            subcarrier_index = 12  # 示例子载波索引
            latest_data = csi_arr[:, subcarrier_index]
            latest_data =latest_data - np.mean(latest_data, axis=0)
            # Compute STFT on the latest data
            f, t, Zxx = stft(latest_data, fs, nperseg=70, noverlap = 69, window = 'hann')
            magnitude = np.abs(Zxx)
            # noise_threshold = 0.4
            # magnitude = np.where(magnitude > noise_threshold, magnitude, 0)
            # Update heatmap data
            stft_heatmap.set_data(magnitude)
            stft_heatmap.set_extent([t.min(), t.max(), f.min(), f.max()])
            stft_heatmap.set_clim(vmin=0, vmax=5)

        return stft_heatmap#, high_freq_line, low_freq_line

    ani = animation.FuncAnimation(fig, update, frames=np.arange(100), interval=update_interval, blit=False)

    plt.show()
# def show_csi_STFT_func(lock, csi_amplitude_array, cache_len, csi_shape, fs=100, update_interval=200):
#     with lock:
#         # Reshape CSI data
#         csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
        
#     # 初始化图表和轴
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
#     plt.subplots_adjust(bottom=0.3, hspace=0.35)

#     # STFT热图轴设置
#     stft_heatmap = ax1.imshow(np.zeros((csi_shape[1]//2 + 1, cache_len)), aspect='auto', origin='lower', cmap='viridis')
#     fig.colorbar(stft_heatmap, ax=ax1, orientation='vertical', label='Magnitude')
#     ax1.set_title('STFT Magnitude Heatmap')
#     ax1.set_xlabel('Time [sec]')
#     ax1.set_ylabel('Frequency [Hz]')

#     # 频率平均值曲线轴设置
#     ax2.set_title('Frequency Averages Over Time')
#     ax2.set_xlabel('Time (s)')
#     ax2.set_ylabel('Magnitude Average')
#     high_freq_line, = ax2.plot([], [], label='High Frequency Avg', color='blue')
#     low_freq_line, = ax2.plot([], [], label='Low Frequency Avg', color='red')
#     ax2.legend()

#     timestamps = []
#     high_freq_averages = []
#     low_freq_averages = []

#     def update(frame):
#         nonlocal timestamps, high_freq_averages, low_freq_averages
#         with lock:
#             # 计算STFT
#             subcarrier_index = 5  # 示例子载波索引
#             latest_data = csi_arr[:, subcarrier_index] - np.mean(csi_arr[:, subcarrier_index])
#             f, t, Zxx = stft(latest_data, fs, nperseg=60, noverlap=58)
#             magnitude = np.abs(Zxx)
#             magnitude = np.where(magnitude > 0.4, magnitude, 0)  # 应用噪声阈值

#             # 更新STFT热图
#             stft_heatmap.set_data(magnitude)
#             stft_heatmap.set_extent([0, magnitude.shape[1], 0, magnitude.shape[0]])

#         return stft_heatmap, high_freq_line, low_freq_line

#     ani = animation.FuncAnimation(fig, update, frames=np.arange(100), interval=update_interval, blit=False)

#     plt.show()