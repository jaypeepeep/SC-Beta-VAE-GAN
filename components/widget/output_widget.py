from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
import os

class OutputWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OutputWidget, self).__init__(parent)
        self.setup_output_collapsible()
        
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
                border: 2px solid #003333;
                color: #003333;
                background-color: transparent;
                font-family: Montserrat;
                font-size: 14px;
                padding: 7px 20px;
                border-radius: 10px;
            }QPushButton:hover {
                background-color: #005555;
                color: #030303; 
            }
        """)
        done_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(done_button)

        download_button = QtWidgets.QPushButton("Download")
        download_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #003333;
                color: #003333;
                background-color: transparent;
                font-family: Montserrat;
                font-size: 14px;
                padding: 7px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #005555; 
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

        sample_output_label = QtWidgets.QLabel("Sample Output")
        sample_output_label.setStyleSheet("""
            font-family: Montserrat;
            font-size: 16px;
            color: black;
            font-weight: bold;
        """)
        sample_output_label.setAlignment(QtCore.Qt.AlignCenter)