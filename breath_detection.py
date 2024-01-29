import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy


def breath_detection_func(lock, breath_lock, csi_amplitude_array, csi_phase_array, csi_shape,
                          breath_detection_data_array, cache_len=100):
    amplitude = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    phase = np.frombuffer(csi_phase_array, dtype=np.float32).reshape(csi_shape)
    value_array = np.frombuffer(breath_detection_data_array, dtype=np.float32).reshape(cache_len)
    last_amplitude = copy.deepcopy(amplitude)
    last_phase = copy.deepcopy(phase)

    i = complex(0, 1)
    pi = 3.1415926535897932384626

    while True:
        if (last_amplitude == amplitude).all():
            continue
        with lock:
            last_amplitude = copy.deepcopy(amplitude)
        if (last_phase == phase).all():
            continue
        with lock:
            last_phase = copy.deepcopy(phase)

        s1, s2 = 49, 15

        a1 = last_amplitude[-1, s1]
        a2 = last_amplitude[-1, s2]
        p1 = last_phase[-1, s1] * pi / 180
        p2 = last_phase[-1, s2] * pi / 180
        subc1 = complex(a1 * np.cos(p1), a1 * np.sin(p1))
        subc2 = complex(a2 * np.cos(p2), a2 * np.sin(p2))

        r = np.angle(subc1 / subc2)

        # print(r)

        with breath_lock:

            value_array[:-1] = value_array[1:]
            value_array[-1] = r

            # print(value_array[-5:])


def breath_plot(breath_lock, breath_detection_data_array, cache_len=100):
    fig, ax = plt.subplots()
    plt.title('Breath Detection')
    plt.xlabel('packets')

    plt.ylabel('Breath wave')
    ax.set_ylim(-1.57, 1.57)

    ax.set_xlim(0, cache_len)
    x = np.arange(0, cache_len, 1)
    detection_data = np.frombuffer(breath_detection_data_array, dtype=np.float32).reshape(cache_len)

    line, = ax.plot(x, detection_data, linewidth=1.0, label='subcarrier')
    plt.legend()

    def init():
        line.set_ydata([np.nan] * len(x))
        return line

    def animate(i):
        with breath_lock:
            # print(detection_data[-5:])
            dec = copy.deepcopy(detection_data)
        line.set_ydata(dec)

        return line

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000 / 25, blit=False, cache_frame_data=False)
    plt.show()