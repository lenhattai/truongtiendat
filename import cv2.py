import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2

class FoodRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.running = False  # Biến cờ để kiểm soát việc chạy camera
        self.model = None  # Khởi tạo model

    def initUI(self):
        self.setWindowTitle('Food Recognition App')
        self.setGeometry(100, 100, 800, 600)

        # Thêm label để hiển thị ảnh 01.jpg
        self.label_logo = QLabel(self)
        self.label_logo.setPixmap(QPixmap('01.jpg'))
        self.label_logo.setAlignment(Qt.AlignCenter)

        # Thêm label để hiển thị ảnh từ camera hoặc file
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)  
        self.label_image.setFixedSize(640, 480) 

        # Thêm label để hiển thị kết quả nhận diện
        self.label_result = QLabel(self)
        self.label_result.setAlignment(Qt.AlignCenter)  
        self.label_result.setFixedSize(640, 50)  

        # Thêm nút truy cập camera
        self.btn_open_camera = QPushButton('Open Camera', self)
        self.btn_open_camera.clicked.connect(self.open_camera)

        # Thêm nút truy cập ảnh
        self.btn_open_image = QPushButton('Open Image', self)
        self.btn_open_image.clicked.connect(self.open_image)

        # Tạo khoảng trống giữa các nút
        spacer = QSpacerItem(100, 120, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Thêm nút vào layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn_open_camera)
        vbox.addWidget(self.btn_open_image)

        # Thêm các thành phần vào layout
        vbox.addWidget(self.label_logo)
        vbox.addWidget(self.label_image)
        vbox.addWidget(self.label_result)

        # Thêm khoảng trống vào layout
        vbox.addItem(spacer)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)
        hbox.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(hbox)
        self.setLayout(layout)

    def open_camera(self):
        # Ẩn cửa sổ hiện tại
        self.hide()
        # Tạo cửa sổ mới để hiển thị camera
        camera_window = CameraWindow(self)
        camera_window.show()

    def open_image(self):
        # Ẩn cửa sổ hiện tại
        self.hide()
        # Tạo cửa sổ mới để chọn ảnh
        image_window = ImageWindow(self)
        image_window.show()

class CameraWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Camera Window')
        self.setGeometry(100, 100, 800, 600)

        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0)

        # Thêm label để hiển thị camera
        self.label_camera = QLabel(self)
        self.label_camera.setAlignment(Qt.AlignCenter)
        self.label_camera.setGeometry(100, 100, 640, 480)

        # Thêm nút Exit
        self.btn_exit = QPushButton('Exit', self)
        self.btn_exit.clicked.connect(self.exit_camera)

        # Thêm layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_camera)
        layout.addWidget(self.btn_exit)
        self.setLayout(layout)

        # Chạy camera
        self.run_camera()

    def run_camera(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = frame_rgb.shape
                bytes_per_line = channels * width
                q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.label_camera.setPixmap(pixmap)
            else:
                break
            cv2.waitKey(20)

    def exit_camera(self):
        self.cap.release()
        self.close()
        self.parent.show()

class ImageWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Window')
        self.setGeometry(100, 100, 800, 600)

        # Thêm label để hiển thị ảnh và kết quả nhận diện
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_result = QLabel(self)
        self.label_result.setAlignment(Qt.AlignCenter)

        # Thêm nút Exit
        self.btn_exit = QPushButton('Exit', self)
        self.btn_exit.clicked.connect(self.exit_image)

        # Thêm layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_image)
        layout.addWidget(self.label_result)
        layout.addWidget(self.btn_exit)
        self.setLayout(layout)

        # Hiển thị cửa sổ chọn ảnh
        self.open_image()

    def open_image(self):
        # Hiển thị cửa sổ chọn ảnh
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if filename:
            image = cv2.imread(filename)
            # Hiển thị ảnh trong label
            q_img = self.convert_cv_qt(image)
            pixmap = QPixmap.fromImage(q_img)
            self.label_image.setPixmap(pixmap)
            # Thực hiện nhận diện và hiển thị kết quả
            class_id, confidence, bbox = self.parent.detect_objects(image)
            if class_id != -1:
                self.label_result.setText(f"Class: {self.parent.class_names[class_id]}, Confidence: {confidence:.2f}")
            else:
                self.label_result.setText("No object detected")

    def convert_cv_qt(self, cv_img):
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return q_img

    def exit_image(self):
        self.close()
        self.parent.show()

def main():
    app = QApplication(sys.argv)
    window = FoodRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()




