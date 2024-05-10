import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QSpacerItem, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
from PIL import Image, ImageTk


class FoodRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.running = False  # Thêm thuộc tính self.running để kiểm soát việc chạy camera

        # Load trained model
        self.model = cv2.dnn.readNetFromTensorflow('D:\\svcode\\111\\model_00.h5')  # Đường dẫn đến file model của bạn

        # Load class names
        self.class_names = ['Bắp cải 24calo/100gr', 'Bí đỏ 76calo/100gr', 'Bí xanh 49calo/100gr','Bông cải xanh 33calo/100gr', 'Bơ động vật 716calo/100gr', 'Cà chua 17calo/100gr', 'Cá hồi 208calo/100gr',
               'Cá ngừ 129calo/100gr','Cà rốt 41calo/100gr','Cá thu 215calo/100gr','Cải thảo 25calo/100gr','Cải xoong 27calo/100gr','Củ cải 14calo/100gr','Dọc mùng 15calo/100gr','Đậu phụ 68calo/100gr',
               'Đùi gà 214calo/100gr','Khoai lang 85calo/100gr','Khoai sọ 64calo/100gr','Khoai tây 76calo/100gr','Mỡ cá 405calo/100gr','Mỡ gà 652calo/100gr','Mỡ heo 895calo/100gr','Nấm kim châm 29calo/100gr',
               'Nấm rơm 22calo/100gr','Phô mai 203calo/100gr','Rau chân vịt 32calo/100gr','Rau mùng tơi 28calo/100gr','Rau muống 30calo/100gr','Sò điệp 111calo/100gr','Sữa tươi 42calo/100gr','Thịt bò 250calo/100gr',
               'Thịt cừu 294calo/100gr','Thịt lợn 216calo/100gr','Tôm 99calo/100gr','Ức gà 129calo/100gr','Váng sữa 375calo/100gr']  # Tên các lớp phù hợp với mô hình của bạn

    def initUI(self):
        self.setWindowTitle('Nhận diện thực phẩm và calo')
        self.setGeometry(100, 100, 800, 600)

        # Thêm label để hiển thị ảnh 01.jpg
        self.label_logo = QLabel(self)
        self.label_logo.setPixmap(QPixmap('01.jpg'))
        self.label_logo.setAlignment(Qt.AlignCenter)
        
        # Thêm label để hiển thị ảnh từ camera hoặc file
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)  # Căn giữa
        self.label_image.setFixedSize(640, 480)  # Đặt kích thước cố định cho label

        # Thêm label để hiển thị kết quả nhận diện
        self.label_result = QLabel(self)
        self.label_result.setAlignment(Qt.AlignCenter)  # Căn giữa
        self.label_result.setFixedSize(640, 50)  # Đặt kích thước cố định cho label

        # Thêm nút mở camera
        self.btn_open_camera = QPushButton('Open Camera', self)
        self.btn_open_camera.clicked.connect(self.toggle_camera)

        # Thêm nút mở ảnh
        self.btn_open_image = QPushButton('Open Image', self)
        self.btn_open_image.clicked.connect(self.open_image)

        # Tạo khoảng trống giữa các nút
        spacer = QSpacerItem(100, 120, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Thêm nút vào layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn_open_camera)
        vbox.addWidget(self.btn_open_image)

        # Thêm các thành phần vào layout
        vbox.addWidget(self.label_image)
        vbox.addWidget(self.label_result)

        # Thêm khoảng trống vào layout
        vbox.addSpacerItem(spacer)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)
        hbox.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(hbox)
        self.setLayout(layout)

    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.run_camera()
            self.btn_open_camera.setText('Stop Camera')  # Thay đổi nút thành "Stop Camera"
        else:
            self.running = False
            self.btn_open_camera.setText('Open Camera')  # Thay đổi nút thành "Open Camera"

    def run_camera(self):
        cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định (index 0)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            class_id, confidence, bbox = self.detect_objects(frame)

            if class_id != -1:
                self.label_result.setText(f"Class: {self.class_names[class_id]}, Confidence: {confidence:.2f}")
            else:
                self.label_result.setText("No object detected")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.label_image.setPixmap(pixmap)
            self.label_image.setScaledContents(True)
            QApplication.processEvents()

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if filename:
            image = cv2.imread(filename)
            class_id, confidence, bbox = self.detect_objects(image)
            if class_id != -1:
                self.label_result.setText(f"Class: {self.class_names[class_id]}, Confidence: {confidence:.2f}")
            else:
                self.label_result.setText("No object detected")

    def detect_objects(self, image):
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.model.setInput(blob)
        output = self.model.forward()

        h, w = image.shape[:2]
        class_id = -1
        confidence = 0
        bbox = []

        for detection in output[0, 0, :, :]:
            score = detection[2]
            if score > confidence:
                class_id = int(detection[1])
                confidence = score
                left = int(detection[3] * w)
                top = int(detection[4] * h)
                right = int(detection[5] * w)
                bottom = int(detection[6] * h)
                bbox = [left, top, right, bottom]

        return class_id, confidence, bbox

def main():
    app = QApplication(sys.argv)
    window = FoodRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


