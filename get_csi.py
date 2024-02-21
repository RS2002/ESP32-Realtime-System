import csv
import serial
import json
from io import StringIO
import numpy as np
from datetime import datetime
from Constant import *
import pymysql


def get_csi(serial_port, csi_amplitude_array, csi_phase_array, csi_shape, lock, host="10.20.14.42", user="zhaozijian", passwd="9213@fCOW", db="wave_data", charset="utf8", chosen_subcarrier = 20, cache_len = 100, store_database=False):
    store=store_database

    csi_amplitude_matrix = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    csi_phase_matrix = np.frombuffer(csi_phase_array, dtype=np.float32).reshape(csi_shape)

    chosen_subcarrier_arr = np.zeros([cache_len])
    csi_data_array = np.zeros([CSI_DATA_LLFT_COLUMNS], dtype=np.complex64)
    current_index = 0

    if store:
        conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)
        cursor = conn.cursor()

    # 读取CSI
    ser = serial.Serial(port=serial_port, baudrate=921600,
                        bytesize=8, parity='N', stopbits=1)
    if ser.isOpen():
        print("open success")
    else:
        print("open failed")
        return
    while True:
        strings = str(ser.readline())
        #print(strings)
        if not strings:
            break
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        index = strings.find('CSI_DATA')
        if index == -1:
            continue
        csv_reader = csv.reader(StringIO(strings))
        csi_data = next(csv_reader)
        if len(csi_data) != len(DATA_COLUMNS_NAMES):
            # print("element number is not equal")
            continue
        try:
            csi_raw_data = json.loads(csi_data[-1])
        except json.JSONDecodeError:
            print(f"data is incomplete")
            continue
        if len(csi_raw_data) != 128 and len(csi_raw_data) != 256 and len(csi_raw_data) != 384:
            print(f"element number is not equal: {len(csi_raw_data)}")
            continue
        # Rotate data to the left
        '''if len(csi_raw_data) == 128:
            csi_vaid_subcarrier_len = CSI_DATA_LLFT_COLUMNS
        else:
            csi_vaid_subcarrier_len = CSI_DATA_COLUMNS'''
        csi_vaid_subcarrier_len = CSI_DATA_LLFT_COLUMNS

        for i in range(csi_vaid_subcarrier_len):
            csi_data_array[i] = complex(csi_raw_data[csi_vaid_subcarrier_index[i] * 2],
                                        csi_raw_data[csi_vaid_subcarrier_index[i] * 2 - 1])


        if store:

            # 存储单一载波的幅度值
            # 每100个csi（约1s）写入一次数据库
            now = datetime.now()
            if current_index == cache_len:
                current_arr_str = ','.join(map(str, chosen_subcarrier_arr))
                sql = "insert into CSI (receive_time,csi_amplitude) values ('" + str(
                    now) + "','" + current_arr_str + "');"
                cursor.execute(sql)
                conn.commit()
                current_index = 0
            else:
                if chosen_subcarrier != -1:
                    chosen_subcarrier_arr[current_index] = np.abs(csi_data_array)[chosen_subcarrier]
                else:
                    chosen_subcarrier_arr[current_index] = np.mean(np.abs(csi_data_array))
                current_index += 1
            # 数据存储8h后删除
            if now.minute == 0:
                hour = (now.hour + 24 - 8) % 24
                sql = "delete from CSI where receive_time like '%" + str(hour) + ":%';"
                # print(sql)
                cursor.execute(sql)
                conn.commit()

            # #存储完整CSI
            # now=datetime.now()
            # current_arr_str=','.join(map(str, csi_data_array))
            # sql="insert into CSI_full (device_num, year, month, day, hour, minute, second, csi) values (0,'"+str(now.year)+"','"+str(now.month)+"','"+str(now.day)+"','"+str(now.hour)+"','"+str(now.minute)+"','"+str(now.second)+"."+"0"*(6-len(str(now.microsecond)))+str(now.microsecond)+"','"+current_arr_str+"');"
            # cursor.execute(sql)
            # conn.commit()
            # # 数据存储8h
            # if now.minute==0:
            #     hour=(now.hour+24-8)%24
            #     sql = "delete from CSI_full where hour="+str(hour)+";"
            #     cursor.execute(sql)
            #     conn.commit()


        # 更新cache
        with lock:
            csi_amplitude_matrix[:-1] = csi_amplitude_matrix[1:]
            csi_amplitude_matrix[-1] = np.abs(csi_data_array)

            csi_phase_matrix[:-1] = csi_phase_matrix[1:]
            csi_phase_matrix[-1] = np.angle(csi_data_array)
            # csi_phase_matrix[-1] = np.unwrap(np.angle(csi_data_array))  # 相位展开并转换回度

            # print(csi_phase_matrix)