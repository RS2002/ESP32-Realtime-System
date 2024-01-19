# ESP32-Realtime-System 开发说明



**开发者：**赵子健、陈廷尉、孟凡一、蔡智捷
**设备：**ESP32-S3 （支持ESP32其他型号）



## 使用说明

首先向ESP32中烧入[esp-csi/examples/get-started/csi_recv_router at master · espressif/esp-csi (github.com)](https://github.com/espressif/esp-csi/tree/master/examples/get-started/csi_recv_router)，并连接至router

之后通过如下命令使用系统

```shell
python main.py --port <port>
```

![](https://raw.githubusercontent.com/RS2002/ESP32-Realtime-System/main/fig/ui.png?token=GHSAT0AAAAAAB7J2R4YJUYGXXOJ5PM4CYHQZNJ7VBQ)

更多参数可使用获取

```shell
python main.py --help
```



**各模块使用说明：TODO**





## 开发说明

**待开发内容：**呼吸检测、跌倒检测、轨迹跟踪

**可使用的变量：**CSI幅度、相位

**使用方法：**

```python
def func(csi_amplitude_array, csi_phase_array, csi_shape, lock): #可根据需要使用csi_amplitude_array和csi_phase_array
    #首先将multiprocessing.RawArray转化为np.array，此步骤无需加锁
    csi_amplitude_matrix = np.frombuffer(csi_amplitude_array, dtype=np.float32).reshape(csi_shape)
    csi_phase_matrix = np.frombuffer(csi_phase_array, dtype=np.float32).reshape(csi_shape)
    
    # CSI不断更新，应该需要再while循环中不断读取
    while True:
        # 之后在读写CSI时需要加锁（这里没有区分读写锁，但各模块不应更改CSI矩阵中数据）
        with lock:
            读取csi_amplitude_matrix/csi_phase_matrix
```

**可用变量说明：**

csi_shape：幅度和相位的shape，大小为“100\*52”，其中100是cache大小，52是载波数（cache大小可通过args修改）
csi_amplitude_array、csi_amplitude_array：形状都为“100\*52”，更新逻辑如下（即最新的数据被添加在array末尾），其中phase是角度制（-180~180）

```python
# 更新cache
with lock:
    csi_amplitude_matrix[:-1] = csi_amplitude_matrix[1:]
    csi_amplitude_matrix[-1] = np.abs(csi_data_array)

    csi_phase_matrix[:-1] = csi_phase_matrix[1:]
    csi_phase_matrix[-1] = np.angle(csi_data_array,deg=True)
```

lock：csi_amplitude_array、csi_amplitude_array的读写锁

**函数接入系统参考：**

![image-20240119125845584](C:\Users\44870\AppData\Roaming\Typora\typora-user-images\image-20240119125845584.png)

