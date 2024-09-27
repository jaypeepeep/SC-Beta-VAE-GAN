from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
import os

class OutputWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OutputWidget, self).__init__(parent)
        self.setup_output_collapsible()

    def get_image_path(self, image_name):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../icon/{image_name}'))

    def setup_output_collapsible(self):
        """Set up the 'Output' collapsible widget"""
        layout = QtWidgets.QVBoxLayout(self)

        file_container = QtWidgets.QWidget(self)
        file_container.setStyleSheet("""
            background-color: #DEDEDE;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 0;
        """)

        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(8)
        shadow_effect.setColor(QColor(0, 0, 0, 160))
        shadow_effect.setOffset(2)
        file_container.setGraphicsEffect(shadow_effect)

        file_container_layout = QtWidgets.QHBoxLayout(file_container)
        file_container_layout.setSpacing(20)
        file_container_layout.setContentsMargins(10, 10, 10, 10)

        file_label = QtWidgets.QLabel("Time-series_Data.zip")
        file_label.setStyleSheet("font-family: Montserrat; font-size: 14px; color: black;")
        file_container_layout.addWidget(file_label)

        button_container = QtWidgets.QWidget(self)
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setSpacing(10)

        done_button = QtWidgets.QPushButton("Done")
        done_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #003333;
                color: #003333;
                background-color: transparent;
                font-family: Montserrat;
                font-size: 14px;
                padding: 7px 20px;
                border-radius: 10px;
            }
        """)
        done_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(done_button)

        download_button = QtWidgets.QPushButton("Download")
        download_button.setStyleSheet("""
            QPushButton {
                background-color: #003333;
                color: white;
                font-family: Montserrat;
                font-size: 14px;
                padding: 7px 20px;
                border-radius: 10px;
            }
        """)
        download_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(download_button)

        delete_button = QtWidgets.QPushButton("X")
        delete_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                font-weight: 600;
                color: #FF5252;
                font-family: Montserrat;
                font-size: 22px;
                padding: 5px;
            }
        """)
        delete_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(delete_button)

        file_container_layout.addWidget(button_container, alignment=QtCore.Qt.AlignRight)

        layout.addWidget(file_container)

        images_outer_container = QtWidgets.QWidget(self)
        images_outer_container.setStyleSheet("background-color: #D9D9D9; border-radius: 0; padding: 15px;")
        images_outer_layout = QtWidgets.QVBoxLayout(images_outer_container)

        sample_output_label = QtWidgets.QLabel("Sample Output")
        sample_output_label.setStyleSheet("""
            font-family: Montserrat;
            font-size: 16px;
            color: black;
            font-weight: bold;
        """)
        sample_output_label.setAlignment(QtCore.Qt.AlignCenter)
        images_outer_layout.addWidget(sample_output_label)

        images_layout = QtWidgets.QHBoxLayout()

        original_container = QtWidgets.QWidget(self)
        original_layout = QtWidgets.QVBoxLayout(original_container)
        original_container.setStyleSheet("background-color: white; padding: 5px; border-radius: 0;")

        original_label = QtWidgets.QLabel("Original Input")
        original_label.setAlignment(QtCore.Qt.AlignCenter)
        original_label.setStyleSheet("font-family: Montserrat; font-size: 12px; color: black;")

        original_image = QtWidgets.QLabel(self)
        original_image_path = self.get_image_path('original_data.png')
        original_qimage = QtGui.QImage(original_image_path)

        if original_qimage.isNull():
            original_image.setText("Image not found.")
        else:
            original_image.setPixmap(QtGui.QPixmap.fromImage(original_qimage).scaled(200, 200, QtCore.Qt.KeepAspectRatio))

        original_image.setAlignment(QtCore.Qt.AlignCenter)

        original_layout.addWidget(original_label)
        original_layout.addWidget(original_image)
        images_layout.addWidget(original_container)

        augmented_container = QtWidgets.QWidget(self)
        augmented_layout = QtWidgets.QVBoxLayout(augmented_container)
        augmented_container.setStyleSheet("background-color: white; padding: 5px; border-radius: 0;")

        augmented_label = QtWidgets.QLabel("Augmented Data")
        augmented_label.setAlignment(QtCore.Qt.AlignCenter)
        augmented_label.setStyleSheet("font-family: Montserrat; font-size: 12px; color: black;")

        augmented_image = QtWidgets.QLabel(self)
        augmented_image_path = self.get_image_path('augmented_data.png')
        augmented_qimage = QtGui.QImage(augmented_image_path)

        if augmented_qimage.isNull():
            augmented_image.setText("Image not found.")
        else:
            augmented_image.setPixmap(QtGui.QPixmap.fromImage(augmented_qimage).scaled(200, 200, QtCore.Qt.KeepAspectRatio))

        augmented_image.setAlignment(QtCore.Qt.AlignCenter)

        augmented_layout.addWidget(augmented_label)
        augmented_layout.addWidget(augmented_image)
        images_layout.addWidget(augmented_container)

        images_outer_layout.addLayout(images_layout)

        layout.addWidget(images_outer_container)
