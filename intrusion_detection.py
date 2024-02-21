import numpy as np
from datetime import datetime
import copy
import pymysql
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# compute the Cosine Similarity of vector A and B
def compute_cos(A,B):
    # compute the abs first (if input is complexity)
    A = np.abs(A)
    B = np.abs(B)
    return np.dot(A,B)/np.sqrt((np.dot(A,A)*np.dot(B,B)))

def intrusion_detection_func(lock,detection_lock, csi_amplitude_array,csi_shape,detection_data_array, threshold=None ,start_len=1000 ,detection_gap=1 ,alarm_interval=3000, chosen_subcarrier=20, method=1,cache_len=100):
    set_threshold = False
    count=0
    alarm_count=0
    csi_arr=np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    detection_data=np.frombuffer(detection_data_array, dtype=np.float32).reshape(cache_len)
    cache_data=None

    while True:
        last_data = copy.deepcopy(cache_data)
        with lock:
            cache_data =csi_arr[:,chosen_subcarrier]
            #cache_data = np.mean(csi_arr,axis=-1) #mean
        if (cache_data==last_data).all():
            continue
        if method == 0:  # According to cosine similarity of CSI
            rate = 0.995
            if threshold.value==-1:
                threshold.value = 1.0
                set_threshold = True
            if np.sum(cache_data[0])!=0:
                if set_threshold and count < start_len:
                    count+=1
                    cos = compute_cos(cache_data[:-detection_gap], cache_data[detection_gap:])
                    with detection_lock:
                        detection_data[:-1] = detection_data[1:]
                        detection_data[-1] = cos
                    if cos <= threshold.value:
                        threshold.value = cos * rate
                else:
                    if count == start_len:
                        count += 1
                        print("Threshold is set as ", threshold.value)
                    if count % detection_gap == 0:
                        cos = compute_cos(cache_data[:-detection_gap], cache_data[detection_gap:])
                        with detection_lock:
                            detection_data[:-1]=detection_data[1:]
                            detection_data[-1]=cos

                        if alarm_count == 0 and cos <= threshold.value:
                            message = "Intrusion detected at " + str(datetime.now())
                            print(message)
                            alarm_count += 1
                        if alarm_count != 0 :
                            alarm_count = (alarm_count + 1) % alarm_interval
        elif method == 1:  # According to range of amplitude
            rate = 1.05
            if threshold.value==-1:
                threshold.value = 0.0
                set_threshold = True
            if np.sum(cache_data[0] )!=0:
                if set_threshold and count < start_len:
                    count+=1
                    ran = np.max(np.abs(cache_data)) - np.min(np.abs(cache_data))
                    # print(ran)
                    with detection_lock:
                        detection_data[:-1] = detection_data[1:]
                        detection_data[-1] = ran
                    if ran >= threshold.value:
                        threshold.value = ran * rate
                else:
                    if count == start_len:
                        count+=1
                        print("Threshold is set: ", threshold.value)
                    if count % detection_gap == 0:
                        ran = np.max(np.abs(cache_data)) - np.min(np.abs(cache_data))
                        with detection_lock:
                            detection_data[:-1]=detection_data[1:]
                            detection_data[-1]=ran

                        if alarm_count == 0 and ran >= threshold.value:
                            message = "Intrusion detected at " + str(datetime.now())
                            print(message)
                            alarm_count += 1
                        if alarm_count != 0 :
                            alarm_count = (alarm_count + 1) % alarm_interval
        elif method == 2:  # According to var of amplitude
            rate = 1.05
            if threshold.value==-1:
                threshold.value = 0.0
                set_threshold = True
            if np.sum(cache_data[0] )!=0:
                if set_threshold and count < start_len:
                    count+=1
                    # ran = np.max(np.abs(cache_data)) - np.min(np.abs(cache_data))
                    ran = np.var(cache_data)
                    # print(ran)
                    with detection_lock:
                        detection_data[:-1] = detection_data[1:]
                        detection_data[-1] = ran
                    if ran >= threshold.value:
                        threshold.value = ran * rate
                else:
                    if count == start_len:
                        count+=1
                        print("Threshold is set: ", threshold.value)
                    if count % detection_gap == 0:
                        # ran = np.max(np.abs(cache_data)) - np.min(np.abs(cache_data))
                        ran = np.var(cache_data)
                        with detection_lock:
                            detection_data[:-1]=detection_data[1:]
                            detection_data[-1]=ran

                        if alarm_count == 0 and ran >= threshold.value:
                            message = "Intrusion detected at " + str(datetime.now())
                            print(message)
                            alarm_count += 1
                        if alarm_count != 0 :
                            alarm_count = (alarm_count + 1) % alarm_interval

def intrusion_history_func(host="10.20.14.42", user="zhaozijian", passwd="9213@fCOW", db="wave_data", charset="utf8"):
    conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)
    cursor = conn.cursor()
    sql="select * from people_action order by time_stamp;"
    cursor.execute(sql)
    result = cursor.fetchall()
    people=[data[1] for data in result]
    plt.plot(people)
    l=len(people)
    ticks=np.linspace(0, l-1, 5)
    labels=[result[int(i)][0] for i in ticks]
    plt.xticks(ticks, labels)
    plt.ylabel("People Num")
    plt.xlabel("Time")
    plt.title("Intrusion Detection History")
    plt.show()

def intrusion_plot(lock,detection_data_array,threshold,method=1,cache_len=100, host="10.20.14.42", user="zhaozijian", passwd="9213@fCOW", db="wave_data", charset="utf8",store_database=False):
    store=store_database
    if store:
        conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)
        cursor = conn.cursor()

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('green')
    plt.title('Intruder Detection')
    plt.xlabel('packets')
    if method == 0:
        plt.ylabel('Cosine Similarity')
        ax.set_ylim(0.8, 1)
    elif method == 1:
        plt.ylabel('Range of Amplitude')
        ax.set_ylim(0, 40)
    elif method == 2:
        plt.ylabel('Variance of Amplitude')
        ax.set_ylim(0, 15)
    ax.set_xlim(0, cache_len)
    x = np.arange(0, cache_len, 1)
    detection_data=np.frombuffer(detection_data_array, dtype=np.float32).reshape(cache_len)


    line, = ax.plot(x, detection_data, linewidth=1.0, label='subcarrier')
    line0, = ax.plot(x, [threshold.value] * cache_len, '--', linewidth=1.0, label='threshold')
    plt.legend()

    def init():
        line.set_ydata([np.nan] * len(x))
        line0.set_ydata([np.nan] * len(x))
        return line, line0

    def animate(i):
        with lock:
            dec=detection_data
            thre=threshold.value
        line.set_ydata(dec)
        line0.set_ydata([thre] * len(x))

        people_num=0
        if thre is not None:
            if method == 0:
                if np.min(dec) <= thre:
                    fig.patch.set_facecolor('red')
                    people_num = 1
                else:
                    fig.patch.set_facecolor('green')
                    people_num = 0
            elif method == 1 or method == 2:
                if np.max(dec) >= thre:
                    fig.patch.set_facecolor('red')
                    people_num = 1
                else:
                    fig.patch.set_facecolor('green')
                    people_num = 0

        if store:
            now = datetime.now()
            sql = "insert into people_action (time_stamp,has_people) values ('" + str(
                now) + "'," + str(people_num) + ");"
            cursor.execute(sql)
            conn.commit()
            # 数据存储8h后删除
            if now.minute == 0:
                hour = (now.hour + 24 - 8) % 24
                sql = "delete from people_action where time_stamp like '%" + str(hour) + ":%';"
                cursor.execute(sql)
                conn.commit()

        return line, line0

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000 / 25, blit=False,cache_frame_data=False)
    plt.show()