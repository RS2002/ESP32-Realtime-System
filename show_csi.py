import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
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
