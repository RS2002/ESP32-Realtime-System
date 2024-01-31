import tkinter as tk
from tkinter import messagebox
from tkinter.font import Font
import argparse
import multiprocessing
import numpy as np
from Constant import *
from get_csi import get_csi
from show_csi import show_csi_func
from intrusion_detection import intrusion_detection_func,intrusion_history_func,intrusion_plot
from gesture_recognition import gesture_recognition,gesture_recognition_plot
from breath_detection import breath_detection_func,breath_plot
from fall_detection import fall_detection_func,fall_plot

process_show_csi=None
process_gesture_classification=None
process_gesture_plot=None
process_intrusion_detection=None
process_intrusion_plot=None
process_intrusion_history=None
process_breath_detection=None
process_breath_plot=None
process_fall_intrusion_detection=None
process_fall_intrusion_plot=None

def get_args():
    parser = argparse.ArgumentParser(description="Read CSI data from serial port")
    parser.add_argument('--port', dest='port', type=str, default="COM6",
                        help="Serial port number of csv_recv device")
    parser.add_argument('--cache_len', dest='cache_len', type=int, default=100,
                        help="Cache size")

    #database
    parser.add_argument("--store_database", action="store_true",default=False,
                        help="Whether to store data in MySQL")
    parser.add_argument('--host', dest='host', type=str, default="10.20.14.42",
                        help="Host of MySQL")
    parser.add_argument('--user', dest='user', type=str, default='zhaozijian',
                        help="Username of MySQL")
    parser.add_argument('--password', dest='passwd', type=str, default='9213@fCOW',
                        help="Password of MySQL")
    parser.add_argument('--database', dest='db', type=str, default='wave_data',
                        help="Which database to use")
    parser.add_argument('--charset', dest='charset', type=str, default='utf8')


    # Intrusion
    parser.add_argument('--chosen_subcarrier', dest='chosen_subcarrier', type=int, default=-1,
                        help="The chosen_subcarrier used for intrusion detection")  # -1表示所有载波的平均值
    parser.add_argument('--threshold', dest='threshold', type=int, default=None,
                        help="Threshold of intrusion detection")
    parser.add_argument('--method', dest='method', type=int, default=2,
                        help="Method of intrusion detection (0: Similarity, 1: Amplitude Range, 2: Amplitude Variance)")
    parser.add_argument('--start_len', dest='start_len', type=int, default=1000,
                        help="Starting number of CSI before detection")
    parser.add_argument('--detection_gap', dest='detection_gap', type=int, default=1)
    parser.add_argument('--alarm_interval', dest='alarm_interval', type=int, default=300)

    # Gesture
    parser.add_argument('--model_path', dest='model_path', type=str, default='./model',help="Gesture Classification Model Path")
    parser.add_argument('--action_class', dest='action_class', type=int, default=6,help="Action class num")
    parser.add_argument('--people_class', dest='people_class', type=int, default=8,help="People class num")

    args = parser.parse_args()
    return args

def show_csi():
    global process_show_csi
    if process_show_csi is None:
        process_show_csi = multiprocessing.Process(target=show_csi_func, args=(lock,csi_amplitude_array,cache_len,csi_shape))
        process_show_csi.start()
    else:
        process_show_csi.kill()
        process_show_csi=None

def gesture_classification():
    global process_gesture_classification,process_gesture_plot
    if process_gesture_classification is None:
        process_gesture_classification=multiprocessing.Process(target=gesture_recognition, args=(lock, csi_amplitude_array, csi_shape, gesture_lock, action_array, people_array, model_path, action_class, people_class))
        process_gesture_classification.start()
        process_gesture_plot=multiprocessing.Process(target=gesture_recognition_plot, args=(action_array, people_array, action_class, people_class))
        process_gesture_plot.start()
    else:
        process_gesture_classification.kill()
        process_gesture_plot.kill()
        process_gesture_classification=None
        process_gesture_plot=None


def intrusion_detection():
    global process_intrusion_detection, process_intrusion_plot
    if process_intrusion_detection is None:
        process_intrusion_detection = multiprocessing.Process(target=intrusion_detection_func, args=(lock, detection_lock, csi_amplitude_array, csi_shape, detection_data_array, threshold ,start_len ,detection_gap ,alarm_interval, chosen_subcarrier, method, cache_len))
        process_intrusion_detection.start()
        process_intrusion_plot = multiprocessing.Process(target=intrusion_plot, args=(detection_lock, detection_data_array, threshold, method, cache_len, args.host, args.user, args.passwd, args.db, args.charset, args.store_database))
        process_intrusion_plot.start()
    else:
        process_intrusion_detection.kill()
        process_intrusion_plot.kill()
        process_intrusion_detection=None
        process_intrusion_plot=None

def intrusion_history():
    global process_intrusion_history
    if process_intrusion_history is None:
        process_intrusion_history = multiprocessing.Process(target=intrusion_history_func, args=(args.host, args.user, args.passwd, args.db, args.charset))
        process_intrusion_history.start()
    else:
        process_intrusion_history.kill()
        process_intrusion_history = None

def trajectory():
    messagebox.showinfo("提示", "待开发模块功能正在开发中")

def breath_detection():
    global process_breath_detection, process_breath_plot
    if process_breath_detection is None:
        process_breath_detection = multiprocessing.Process(target=breath_detection_func, args=(
        lock, breath_lock, csi_amplitude_array, csi_phase_array,csi_shape,breath_detection_data_array,cache_len))
        process_breath_detection.start()
        process_breath_plot = multiprocessing.Process(target=breath_plot, args=(breath_lock, breath_detection_data_array, cache_len))
        process_breath_plot.start()
    else:
        process_breath_detection.kill()
        process_breath_plot.kill()
        process_breath_detection = None
        process_breath_plot = None

def fall_detection():
    global process_fall_intrusion_detection, process_fall_intrusion_plot
    fall_method=2
    fall_threshold=None
    if process_fall_intrusion_detection is None:
        process_fall_intrusion_detection = multiprocessing.Process(target=fall_detection_func, args=(
        lock, fall_detection_lock, csi_amplitude_array, csi_shape, fall_detection_data_array, fall_threshold, start_len, detection_gap,
        alarm_interval, chosen_subcarrier, fall_method, cache_len))
        process_fall_intrusion_detection.start()
        process_fall_intrusion_plot = multiprocessing.Process(target=fall_plot, args=(
        fall_detection_lock, fall_detection_data_array, fall_threshold, fall_method, cache_len, args.host, args.user, args.passwd, args.db,
        args.charset, args.store_database))
        process_fall_intrusion_plot.start()
    else:
        process_fall_intrusion_detection.kill()
        process_fall_intrusion_plot.kill()
        process_fall_intrusion_detection = None
        process_fall_intrusion_plot = None
def future_module():
    messagebox.showinfo("提示", "待开发模块功能正在开发中")


if __name__ == '__main__':
    # 多进程部分
    lock = multiprocessing.Lock() # CSI锁
    detection_lock = multiprocessing.Lock() # 入侵检测中间结果锁
    gesture_lock = multiprocessing.Lock() # 手势识别结果锁
    breath_lock = multiprocessing.Lock() # 呼吸检测结果锁
    fall_detection_lock = multiprocessing.Lock() # 跌倒检测中间结果锁


    # Common
    args = get_args()
    cache_len = args.cache_len
    csi_shape = (cache_len, CSI_DATA_LLFT_COLUMNS)
    csi_amplitude_array = multiprocessing.RawArray('f', np.zeros(csi_shape, dtype=np.float32).ravel())  # CSI幅度矩阵
    csi_amplitude_matrix = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    csi_phase_array = multiprocessing.RawArray('f', np.zeros(csi_shape, dtype=np.float32).ravel())  # CSI相位矩阵
    csi_phase_matrix = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)

    # Intrusion
    chosen_subcarrier = args.chosen_subcarrier
    start_len = args.start_len
    detection_gap = args.detection_gap
    alarm_interval = args.alarm_interval
    method = args.method
    threshold = args.threshold if args.threshold is not None else -1
    threshold = multiprocessing.Value('f', threshold)  # 入侵检测阈值
    detection_data_array = multiprocessing.RawArray('f', np.zeros(cache_len, dtype=np.float32).ravel())  # 入侵检测中间结果
    detection_data_matrix = np.frombuffer(detection_data_array, dtype=np.float32).reshape(cache_len)

    # Fall
    fall_threshold = args.threshold if args.threshold is not None else -1
    fall_threshold = multiprocessing.Value('f', fall_threshold)  # 跌倒检测阈值
    fall_detection_data_array = multiprocessing.RawArray('f', np.zeros(cache_len, dtype=np.float32).ravel())  # 跌倒检测中间结果
    fall_detection_data_matrix = np.frombuffer(fall_detection_data_array, dtype=np.float32).reshape(cache_len)

    # Gesture
    model_path=args.model_path # 手势识别模型路径
    action_class=args.action_class # 手势数目
    people_class=args.people_class # 人员数目
    action_array = multiprocessing.RawArray('f', np.zeros(action_class, dtype=np.float32).ravel())  # 动作分类结果
    action_matrix = np.frombuffer(action_array, dtype=np.float32).reshape(action_class)
    people_array = multiprocessing.RawArray('f', np.zeros(people_class, dtype=np.float32).ravel())  # 人员分类结果
    people_matrix = np.frombuffer(people_array, dtype=np.float32).reshape(people_class)

    # Breath
    breath_detection_data_array = multiprocessing.RawArray('f', np.zeros(cache_len, dtype=np.float32).ravel())  # 呼吸检测结果
    breath_detection_data_matrix = np.frombuffer(breath_detection_data_array, dtype=np.float32).reshape(cache_len)

    # Start Read CSI
    process_get_csi = multiprocessing.Process(target=get_csi, args=(args.port, csi_amplitude_array, csi_phase_array, csi_shape, lock, args.host, args.user, args.passwd, args.db, args.charset, chosen_subcarrier, cache_len, args.store_database))
    process_get_csi.start()





    # UI部分
    # 创建主窗口
    root = tk.Tk()
    root.title("Wi-Fi感知实时演示")
    root.geometry("300x250")

    # 创建一个Frame作为标题区域
    title_frame = tk.Frame(root)
    title_frame.pack(pady=20)

    # 创建标题字体
    title_font = Font(family="楷体", size=20)

    # 创建标题标签
    title_label = tk.Label(title_frame, text="实时Wi-Fi感知系统")
    title_label.pack()

    # 创建按钮框架
    button_frame = tk.Frame(root)
    button_frame.pack()

    # 创建显示CSI按钮，绑定对应的函数
    btn_show_csi = tk.Button(button_frame, text="显示CSI", command=show_csi, font=("Helvetica", 12))
    btn_show_csi.grid(row=1, column=0, padx=10, pady=10)

    # 创建其它按钮，绑定对应的函数
    # btn_locate_track = tk.Button(button_frame, text="手势识别", command=gesture_classification, font=("Helvetica", 12))
    # btn_locate_track.grid(row=1, column=1, padx=10, pady=10)

    btn_locate_track = tk.Button(button_frame, text="呼吸检测", command=breath_detection, font=("Helvetica", 12))
    btn_locate_track.grid(row=1, column=1, padx=10, pady=10)

    btn_intrusion_detection = tk.Button(button_frame, text="入侵检测", command=intrusion_detection, font=("Helvetica", 12))
    btn_intrusion_detection.grid(row=2, column=0, padx=10, pady=10)

    # btn_future_module = tk.Button(button_frame, text="入侵检测历史分析", command=intrusion_history, font=("Helvetica", 12))
    # btn_future_module.grid(row=2, column=1, padx=10, pady=10)

    btn_future_module = tk.Button(button_frame, text="跌倒检测", command=fall_detection, font=("Helvetica", 12))
    btn_future_module.grid(row=2, column=1, padx=10, pady=10)

    btn_future_module = tk.Button(button_frame, text="轨迹跟踪", command=trajectory, font=("Helvetica", 12))
    btn_future_module.grid(row=3, column=0, padx=10, pady=10)

    btn_future_module = tk.Button(button_frame, text="待开发模块", command=future_module, font=("Helvetica", 12))
    btn_future_module.grid(row=3, column=1, padx=10, pady=10)

    # 运行主循环
    root.mainloop()

