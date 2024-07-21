import cv2
import numpy as np

def get_dst_points():
    print("请在图片中依次点击四个锚点")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    # 存储点击的点的数组
    points = []

    # 鼠标点击事件的回调函数
    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            # 当点击了四个点后，退出
            if len(points) == 4:
                cv2.destroyAllWindows()
                cap.release()

    # 创建窗口并设置鼠标回调函数
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_click)

    while True:
        # 读取摄像头图像
        ret, frame = cap.read()
        if not ret:
            break

        # 显示已经点击的点
        for i, point in enumerate(points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{i + 1}", (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow("Image", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 打印存储的点
    print("锚点坐标采集完成:", points)
    return np.array(points, dtype="float32")

def LoFi(src_points = np.array([[0, 0], [1.80, 0], [0, 4.80], [1.80, 4.80]], dtype="float32"),dst_points = None, model_path = "./yolo"):
    if dst_points is None:
        dst_points = get_dst_points()
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    def draw_text(img, text, pos, font, font_scale, font_thickness, text_color, text_color_bg):
        """
        Draw multiline text on an image.
        """
        x, y = pos
        for i, line in enumerate(text.split('\n')):
            (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            cv2.rectangle(img, (x, y - h - 5), (x + w, y + 5), text_color_bg, -1)
            cv2.putText(img, line, (x, y), font, font_scale, text_color, font_thickness)
            y += h + 10  # Line spacing

    # 加载 YOLO
    net = cv2.dnn.readNet(model_path+"/yolov3.weights", model_path+"/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # with open("coco.names", "r") as f:
    #     classes = [line.strip() for line in f.readlines()]

    # 启动摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        height, width, channels = frame.shape

        # 预处理
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 分析输出
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # 框坐标
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 框起始点
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 非极大值抑制
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        title = "No object detected"
        # 只保留置信度最高的框
        if len(indexes) > 0:
            best_index = indexes.flatten()[0]
            x, y, w, h = boxes[best_index]
            # label = str(classes[class_ids[best_index]])
            confidence = confidences[best_index]

            # 计算底部中点坐标
            bottom_center_x = x + w // 2
            bottom_center_y = y + h

            foot_position_image = (bottom_center_x, bottom_center_y)
            person_img_coords = np.array([[foot_position_image[0], foot_position_image[1]]],
                                         dtype="float32")
            actual_coords = cv2.perspectiveTransform(np.array([person_img_coords]), np.linalg.inv(M))

            title = f"Coordinate:({actual_coords[0, 0, 0]:.2f}, {actual_coords[0, 0, 1]:.2f}), Confidence Level:{confidence:.2f}"
            # title = f"Coordinate:({bottom_center_x}, {bottom_center_y}), Confidence:{confidence:.2f}"
            # 画框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 在图像上方添加标题
        frame = cv2.copyMakeBorder(frame, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        draw_text(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, (255, 255, 255), (0, 0, 0))

        # 显示实时视频流
        cv2.imshow("Image", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#test
if __name__ == '__main__':
    LoFi()
