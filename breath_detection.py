import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy


def breath_detection_func(lock, breath_lock, csi_amplitude_array, csi_phase_array,csi_shape,breath_detection_data_array,cache_len=100):
    amplitude=np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    phase=np.frombuffer(csi_phase_array, dtype=np.float32).reshape(csi_shape)
    value_array=np.frombuffer(breath_detection_data_array, dtype=np.float32).reshape(cache_len)
    last_amplitude=copy.deepcopy(amplitude)

    while True:
        if (last_amplitude==amplitude).all():
            continue
        with lock:
            last_amplitude = copy.deepcopy(amplitude)
        #TODO: 添加代码，数据存储到value_array中



def breath_plot(lock, breath_detection_data_array, cache_len=100):

    fig, ax = plt.subplots()
    plt.title('Breath Detection')
    plt.xlabel('packets')

    #TODO: 设置y轴信息
    plt.ylabel('')
    ax.set_ylim(0.8, 1)

    ax.set_xlim(0, cache_len)
    x = np.arange(0, cache_len, 1)
    detection_data=np.frombuffer(breath_detection_data_array, dtype=np.float32).reshape(cache_len)



    line, = ax.plot(x, detection_data, linewidth=1.0, label='subcarrier')
    plt.legend()

    def init():
        line.set_ydata([np.nan] * len(x))
        return line

    def animate(i):
        with lock:
            dec=copy.deepcopy(detection_data)
        line.set_ydata(dec)

        return line

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000 / 25, blit=False,cache_frame_data=False)
    plt.show()