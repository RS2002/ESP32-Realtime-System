import time
import numpy as np
from datetime import datetime
import copy
import pymysql
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import stft

# compute the Cosine Similarity of vector A and B
def compute_cos(A,B):
    # compute the abs first (if input is complexity)
    A = np.abs(A)
    B = np.abs(B)
    return np.dot(A,B)/np.sqrt((np.dot(A,A)*np.dot(B,B)))
def compute_stft(csi_data, fs, nperseg, noverlap):
    f, t, Zxx = stft(csi_data, fs, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    return f, t, magnitude
def fall_detection_func(lock, detection_lock, csi_amplitude_array, csi_shape, detection_data_array, 
                        threshold1=2, threshold2=3,chosen_subcarrier=5, fs=100, stft_nperseg=30, stft_noverlap=28, high_freq_threshold=20, 
                        low_freq_threshold=5, duration=5,check_interval = 1):
    csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    count = 0  # Initialize count for low frequency condition duration
    while True:
        with lock:
            latest_data = csi_arr[:, chosen_subcarrier]

        f, t, Zxx = stft(latest_data, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
        magnitude = np.abs(Zxx)
        print('最小')
        print(np.min(magnitude))
        print('最大')
        print(np.max(magnitude))
        # 检查高频条件...
        if detection_data_array[-1] == 0:
            # 如果检测到高频条件，更新状态为1，并记录当前时间
            if np.sum(magnitude[:,high_freq_threshold:] > threshold1)>80:
                print(np.sum(magnitude[:,high_freq_threshold:] > threshold1))
                with detection_lock:
                    detection_data_array[-1] = 1
                    possible_fall_start_time = time.time()

        elif detection_data_array[-1] == 1:
            # 如果处于可能跌倒状态，检查是否满足持续时间条件
            if possible_fall_start_time and ((time.time() - possible_fall_start_time) <= duration):
                # low_freq_condition = np.all(magnitude[f < low_freq_threshold, :] < threshold2, axis=0)
                print(np.sum(magnitude[:,:low_freq_threshold]< threshold2))
                if np.sum(magnitude[:,:low_freq_threshold] < threshold2)<50:
                    # print(np.sum(magnitude[:,:low_freq_threshold]))
                    count += 1
                    print(count)
                    # 如果低频条件在至少5秒的时间里被满足，更新状态为跌倒
                    if count >= 5000:
                        with detection_lock:
                            detection_data_array[-1] = 2  # 更新状态为跌倒
                            count = 0
            else:
                # 如果超过了检测时间窗口，重置状态
                with detection_lock:
                    detection_data_array[-1] = 0
                    count = 0
                    possible_fall_start_time = None

        # 休眠一段时间再次检查
        # time.sleep(check_interval)
    # while True:
    #     with lock:
    #         latest_data = csi_arr[:, chosen_subcarrier]

    #     f, t, Zxx = stft(latest_data, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
    #     magnitude = np.abs(Zxx)
        
    #     # Check for high frequency condition
    #     if detection_data_array[-1] < 1:
    #         high_freq_condition = np.any(magnitude[:, np.newaxis, high_freq_threshold:] > threshold1, axis=0)
    #         if np.any(high_freq_condition):
    #             with detection_lock:
    #                 detection_data_array[-1] = 1  # Possible fall start detected
    #                 count = 0  # Reset count when new high frequency condition is detected

    #     # Check for low frequency condition if a possible fall start was detected
    #     elif detection_data_array[-1] == 1:
    #         low_freq_condition = np.all(magnitude[:, np.newaxis, :low_freq_threshold] < threshold2, axis=0)
    #         if np.all(low_freq_condition):
    #             count += 1
    #             if count >= (duration * fs):  # Check if the condition is met continuously for 'duration' seconds
    #                 with detection_lock:
    #                     detection_data_array[-1] = 2  # Fall detected
    #                     count = 0  # Reset count after detection
    #         else:

    #             count = 0  # Reset count if condition is not continuously met


    #     # Reset to no fall after alarm_interval or if the condition for fall detection is not met
    #     if detection_data_array[-1] == 2 or count == 0:
    #         time.sleep(alarm_interval / 1000.0)  # Sleep for the alarm_interval before next check
    #         with detection_lock:
    #             detection_data_array[-1] = 0  # Reset to no fall
    #             count = 0

# def fall_detection_func(lock, detection_lock, csi_amplitude_array, csi_shape, detection_data_array, 
#                         threshold1=2, threshold2=2, alarm_interval=3000, 
#                         chosen_subcarrier=5, fs=100, stft_nperseg=30, stft_noverlap=28, high_freq_threshold=15, 
#                         low_freq_threshold=5, duration=10, check_interval=1):
#     csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
#     detection_status = 0  # Initialize detection status as no fall

#     while True:
#         with lock:
#             latest_data = csi_arr[:, chosen_subcarrier]

#         f, t, Zxx = stft(latest_data, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
#         magnitude = np.abs(Zxx)
#         # 使用 np.newaxis 来增加一个维度，以便正确广播

#         if detection_data_array[-1]<1:
#             if np.any((magnitude[:,high_freq_threshold:] > threshold1)):
#                 with detection_lock:
#                     detection_data_array[-1]=1
#         elif detection_data_array[-1]==1:
#             time.sleep(1)
#             if np.any((magnitude[:,low_freq_threshold:] < threshold1)):
#                 count = count +1

#         # 现在，我们可以检查每个时间点是否有任何高于阈值的高频幅度
        
#             detection_status = 1  # Possible fall start detected

#             # Initialize counter to measure duration where low frequency magnitude is below threshold2
#             low_freq_below_threshold_duration = 0


#             # Check low frequency condition over the next 'duration' seconds
#             for offset in range(0, duration * fs, check_interval * fs):
#                 _, _, Zxx_subsequent = stft(latest_data[offset:], fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
#                 magnitude_subsequent = np.abs(Zxx_subsequent)
#                 print(magnitude_subsequent.shape)
#                 low_freq_condition = np.all(magnitude_subsequent[f < low_freq_threshold, :] < threshold2)
                
#                 if low_freq_condition:
#                     low_freq_below_threshold_duration += check_interval
#                     if low_freq_below_threshold_duration >= 5:  # If condition met for at least 5 seconds
#                         detection_status = 2  # Fall detected
#                         break
#                 else:
#                     low_freq_below_threshold_duration = 0  # Reset if condition not continuously met

#             with detection_lock:
#                 # Update the shared detection status
#                 detection_data_array[-1] = detection_status

#             if detection_status == 2:
#                 print("Fall detected.")
#             else:
#                 detection_status = 0  # Reset detection status if fall not confirmed

#         time.sleep(alarm_interval / 1000.0)  # Sleep between checks, adjust as necessary

# def fall_detection_func(lock, detection_lock, csi_amplitude_array, csi_shape, detection_data_array, 
#                         threshold1=2, threshold2=2, chosen_subcarrier=5, fs=100, stft_nperseg=30, stft_noverlap=28, 
#                         high_freq_threshold=20, low_freq_threshold=5):
#     csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    
#     while True:
#         with lock:
#             latest_data = csi_arr[:, chosen_subcarrier]

#         f, t, Zxx = stft(latest_data, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
#         magnitude = np.abs(Zxx)
        
#         # Identify peaks above threshold1 for frequencies higher than high_freq_threshold
#         # high_freq_peaks = np.where((magnitude > threshold1) & (f > high_freq_threshold))
#         high_freq_peaks = np.where((magnitude > threshold1) & (f[:, None] > high_freq_threshold))

        
#         if high_freq_peaks[0].size > 0:
#             print(high_freq_peaks)
#             for peak_index in np.nditer(high_freq_peaks[0]):
#                 peak_time = t[peak_index]
#                 subsequent_time_indices = (t >= peak_time) & (t <= peak_time + 10)
                
#                 # Check for low frequency condition in the next 10 seconds
#                 low_freq_magnitude = magnitude[f < low_freq_threshold, :][:, subsequent_time_indices]
                
#                 # Calculate the total time where magnitude stays below threshold2
#                 below_threshold_time = np.sum(np.all(low_freq_magnitude < threshold2, axis=0)) / fs
                
#                 # Update the detection status based on the condition
#                 if below_threshold_time >= 5:
#                     detection_status = 2  # Fall detected
#                 else:
#                     detection_status = 0  # No fall

#                 with detection_lock:
#                     # Assuming detection_data stores the latest status at the last index
#                     detection_data_array[-1] = detection_status
#                     break  # Break after processing the first peak satisfying the condition

#         time.sleep(1)  # Adjust based on the required checking frequency
# def fall_detection_func(lock, detection_lock, csi_amplitude_array, csi_shape, detection_data_array, 
#                         threshold=2, alarm_interval=3000, 
#                         chosen_subcarrier=20, cache_len=100, fs=100, stft_nperseg=30, 
#                         stft_noverlap=28, freq_threshold=15):

#     alarm_count = 0
#     csi_arr = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
#     detection_data = np.frombuffer(detection_data_array, dtype=np.float32).reshape((cache_len, 2))  # Assuming 2D array for time and status
#     # detection_data = np.frombuffer(detection_data_array, dtype=np.float32).reshape((cache_len, 2))

#     while True:
#         with lock:
#             latest_data = csi_arr[:, chosen_subcarrier]

#         f, t, Zxx = stft(latest_data, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
#         magnitude = np.abs(Zxx)
        
#         # Identify peaks that exceed the threshold
#         peaks_indices = np.where((magnitude > threshold) & (f[:, None] > freq_threshold))
#         print(peaks_indices)
#         if peaks_indices[0].size > 0:
#             print(111)
#             fall_detected = False
#             # Analyze the signal after each peak to detect a fall
#             for peak_index in np.nditer(peaks_indices[0]):
#                 start_time_index = np.searchsorted(t, t[peak_index] + 10)  # 10 seconds after the peak
#                 if start_time_index < len(t):
#                     subsequent_magnitude = magnitude[:, start_time_index:]
#                     # Check if the mean magnitude stays below the threshold for more than 3 seconds
#                     if np.all(subsequent_magnitude.max(axis=0) < threshold):
#                         fall_detected = True
#                         break

#             with detection_lock:
#                 # Update the detection status: 0 for no fall, 1 for potential fall, 2 for fall detected
#                 detection_status = 2 if fall_detected else 1  # Assuming fall is detected for simplification
#                 detection_data[-1, 0] = datetime.now().timestamp()  # Update time
#                 detection_data[-1, 1] = detection_status  # Update status

#             if fall_detected:
#                 print(f"Fall detected at {datetime.now()}")
#                 alarm_count += 1
#                 # Reset after detection
#                 if alarm_count >= alarm_interval:
#                     alarm_count = 0
#         time.sleep(1)  # Check every second, adjust as needed


def fall_plot(lock, detection_data_array, cache_len=100, update_interval=100):
    fig, ax = plt.subplots()
    plt.title('Fall Detection Status')
    plt.xlabel('Time')
    plt.ylabel('Status')
    status_bar = ax.bar(['Fall Risk'], [0], color='green')
    ax.set_ylim(0, 2)  # Set the limit to include the "Fall Detected" status
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['No Fall', 'Possible Fall Start', 'Fall Detected'])

    def update_status(frame):
        with lock:
            # 正确地访问共享数组，并重塑为二维形状
            detection_data = np.frombuffer(detection_data_array, dtype=np.float32).reshape((cache_len, 2))
            # 更新绘图逻辑以反映二维数据的使用
            # 示例代码省略了具体的绘图逻辑

            latest_status = detection_data[-1, 1]  # The last element contains the latest status
            # print(latest_status)
            # Update the bar based on the latest detection status
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
# def fall_plot(lock,detection_data_array,threshold,method=1,cache_len=100, host="10.20.14.42", user="zhaozijian", passwd="9213@fCOW", db="wave_data", charset="utf8",store_database=False):
#     store=store_database
#     if store:
#         conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)
#         cursor = conn.cursor()

#     fig, ax = plt.subplots()
#     fig.patch.set_facecolor('green')
#     plt.title('Fall Detection')
#     plt.xlabel('packets')
#     if method == 0:
#         plt.ylabel('Cosine Similarity')
#         ax.set_ylim(0.7, 1)
#     elif method == 1:
#         plt.ylabel('Range of Amplitude')
#         ax.set_ylim(0, 40)
#     elif method == 2:
#         plt.ylabel('Variance of Amplitude')
#         ax.set_ylim(0, 50)
#     ax.set_xlim(0, cache_len)
#     x = np.arange(0, cache_len, 1)
#     detection_data=np.frombuffer(detection_data_array, dtype=np.float32).reshape(cache_len)


#     line, = ax.plot(x, detection_data, linewidth=1.0, label='subcarrier')
#     line0, = ax.plot(x, [threshold.value] * cache_len, '--', linewidth=1.0, label='threshold')
#     plt.legend()

#     def init():
#         line.set_ydata([np.nan] * len(x))
#         line0.set_ydata([np.nan] * len(x))
#         return line, line0

#     def animate(i):
#         with lock:
#             dec=detection_data
#             thre=threshold.value
#         line.set_ydata(dec)
#         line0.set_ydata([thre] * len(x))

#         people_num=0
#         if thre is not None:
#             if method == 0:
#                 if np.min(dec) <= thre:
#                     fig.patch.set_facecolor('red')
#                     people_num = 1
#                 else:
#                     fig.patch.set_facecolor('green')
#                     people_num = 0
#             elif method == 1 or method == 2:
#                 if np.max(dec) >= thre:
#                     fig.patch.set_facecolor('red')
#                     people_num = 1
#                 else:
#                     fig.patch.set_facecolor('green')
#                     people_num = 0

#         if store:
#             now = datetime.now()
#             sql = "insert into people_action (time_stamp,has_people) values ('" + str(
#                 now) + "'," + str(people_num) + ");"
#             cursor.execute(sql)
#             conn.commit()
#             # 数据存储8h后删除
#             if now.minute == 0:
#                 hour = (now.hour + 24 - 8) % 24
#                 sql = "delete from people_action where time_stamp like '%" + str(hour) + ":%';"
#                 cursor.execute(sql)
#                 conn.commit()

#         return line, line0

#     ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000 / 25, blit=False,cache_frame_data=False)
#     plt.show()




