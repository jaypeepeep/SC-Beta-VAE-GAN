from PyQt5 import QtWidgets, QtGui, QtCore
import os
import shutil

class FileContainerWidget(QtWidgets.QWidget):
    remove_file_signal = QtCore.pyqtSignal(str) 
    def __init__(self, file_path, parent=None):
        super(FileContainerWidget, self).__init__(parent)
        self.file_path = file_path
        self.file_name = file_path
        self.setupUi()

    def setupUi(self):
        # Set the background color for the entire widget
        self.setStyleSheet("background: #DEDEDE;")
        
        # Create a container widget for the layout
        self.container = QtWidgets.QWidget(self)
        self.container.setStyleSheet("background: #DEDEDE;")
        self.container.setContentsMargins(0, 0, 0, 0)
        
        # Set layout for the container
        self.layout = QtWidgets.QHBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)  # Ensure no spacing between items

        # Label to display the file name
        self.file_label = QtWidgets.QLabel(self.file_name, self.container)
        self.file_label.setStyleSheet(" margin-left: 10px; background: #DEDEDE; padding: 5px; color: black; color: #000; font-family: Montserrat; font-size: 14px; font-style: normal; font-weight: 600; line-height: normal;") 
        self.layout.addWidget(self.file_label)
        
        # Spacer to push buttons to the right
        self.spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layout.addItem(self.spacer)
        
        # Button to remove the file
        self.remove_button = QtWidgets.QPushButton(self.container)
        self.remove_button.setIcon(QtGui.QIcon(self.get_image_path('close.png')))
        self.remove_button.setIconSize(QtCore.QSize(20, 20))
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
background-color: #003333; color: white; font-family: Montserrat; font-size: 14px; font-weight: 600; padding: 8px 16px; border-radius: 5px;
            }
            QPushButton:hover {
                background: #C0C0C0;
            }
        """)
        self.download_button.clicked.connect(self.download_file)
        self.layout.addWidget(self.download_button)
        
        # Button to retry an action
        self.retry_button = QtWidgets.QPushButton(self.container)
        self.retry_button.setIcon(QtGui.QIcon(self.get_image_path('restart.png')))
        self.retry_button.setIconSize(QtCore.QSize(42, 78))
        self.retry_button.setStyleSheet("""
            QPushButton {
                margin: 10px;
                width: 32px;
                height: 59px;
                background: #DEDEDE;
                border: none;
            }
            QPushButton:hover {
                background: #C0C0C0;
            }
        """)
        self.retry_button.clicked.connect(self.retry_action)
        self.layout.addWidget(self.retry_button)
        
        # Set the container as the central widget of this class
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.addWidget(self.container)
        self.setLayout(self.main_layout)

    def get_image_path(self, image_name):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../icon/{image_name}'))

    def remove_file(self):
        # This method removes the widget from its parent
        self.setParent(None)
        self.remove_file_signal.emit(self.file_path)  # Emit signal when removed
        self.deleteLater() 

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

    def retry_action(self):
        # Implement the retry logic here
        print("Retry button clicked")

    def hide_remove_button(self):
        # Hide the remove button
        self.remove_button.setVisible(False)

    def hide_download_button(self):
        # Hide the download button
        self.download_button.setVisible(False)

    def hide_retry_button(self):
        # Hide the retry button
        self.retry_button.setVisible(False)
