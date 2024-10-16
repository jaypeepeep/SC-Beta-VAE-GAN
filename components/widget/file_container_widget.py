from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
import os
import shutil

class FileContainerWidget(QtWidgets.QWidget):
    remove_file_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, file_path, parent=None):
        super(FileContainerWidget, self).__init__(parent)
        self.file_path = file_path
        self.file_name = os.path.basename(self.file_path)
        self.setupUi()

    def setupUi(self):
        # Set the background color for the entire widget
        self.setStyleSheet("background: #DEDEDE;")

        # Create a container widget for the layout
        self.container = QtWidgets.QWidget(self)
        self.container.setStyleSheet("background: #DEDEDE; border-radius: 0;")
        self.container.setContentsMargins(0, 0, 0, 0)

        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(8)
        shadow_effect.setColor(QColor(0, 0, 0, 160))
        shadow_effect.setOffset(2)
        self.container.setGraphicsEffect(shadow_effect)

        # Set layout for the container
        self.layout = QtWidgets.QHBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)  # Ensure no spacing between items

        # Label to display the file name
        self.file_label = QtWidgets.QLabel(self.file_name, self.container)
        self.file_label.setStyleSheet(" margin-left: 10px; background: #DEDEDE; padding: 5px; color: black; color: #000; font-family: Montserrat; font-size: 14px;")
        self.layout.addWidget(self.file_label)

        # Spacer to push buttons to the right
        self.spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layout.addItem(self.spacer)

        # Button to remove the file
        self.remove_button = QtWidgets.QPushButton(self.container)
        self.remove_button.setIcon(QtGui.QIcon(self.get_image_path('close.png')))
        self.remove_button.setIconSize(QtCore.QSize(42, 78))
        self.remove_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.remove_button.setStyleSheet(""" 
            QPushButton {
                margin: 1px;
                background: #DEDEDE;
                border: none;
            }
            QPushButton:hover {
                background: #C0C0C0;
            }
        """)
        self.remove_button.clicked.connect(self.remove_file)
        self.layout.addWidget(self.remove_button)

        # Button to download the file
        self.download_button = QtWidgets.QPushButton("Download", self.container)
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: #003333; 
                color: white; font-family: 
                Montserrat; font-size: 14px; 
                font-weight: 600; 
                padding: 8px 16px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #C0C0C0;
            }
        """)
        self.download_button.clicked.connect(self.download_file)
        self.layout.addWidget(self.download_button)

        # Set the container as the central widget of this class
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.addWidget(self.container)
        self.setLayout(self.main_layout)

    def get_image_path(self, image_name):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../icon/{image_name}'))

    def remove_file(self):
        """Emit signal to remove the file from the UI and uploaded_files list."""
        self.remove_file_signal.emit(self.file_path, self.file_name)
        self.deleteLater()  # Schedule this widget for deletion

    def download_file(self):
        # Create a QFileDialog to prompt the user for a save location
        options = QtWidgets.QFileDialog.Options()
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", self.file_name, "All Files (*)", options=options)
        
        if save_path:
            try:
                # Define the source path in the uploads folder
                source_path = os.path.join(os.path.dirname(__file__), '../../uploads', self.file_name)
                
                # Copy the file to the specified save path
                shutil.copy(source_path, save_path)
                
                QtWidgets.QMessageBox.information(self, "Success", "File downloaded successfully!")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to download file:\n{e}")

    def hide_remove_button(self):
        # Hide the remove button
        self.remove_button.setVisible(False)

    def hide_download_button(self):
        # Hide the download button
        self.download_button.setVisible(False)