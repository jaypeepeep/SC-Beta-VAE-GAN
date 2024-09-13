from PyQt5 import QtWidgets, QtGui, QtCore
import os

class FileContainerWidget(QtWidgets.QWidget):
    def __init__(self, file_name, parent=None):
        super(FileContainerWidget, self).__init__(parent)
        self.file_name = file_name
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
        self.file_label.setStyleSheet("background: #DEDEDE; padding: 5px; color: black;")  # Set background color and text color
        self.layout.addWidget(self.file_label)
        
        # Spacer to push buttons to the right
        self.spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layout.addItem(self.spacer)
        
        # Button to remove the file
        self.remove_button = QtWidgets.QPushButton(self.container)
        self.remove_button.setIcon(QtGui.QIcon(self.get_image_path('close.png')))
        self.remove_button.setIconSize(QtCore.QSize(30, 30))
        self.remove_button.setStyleSheet("""
            QPushButton {
                margin: 10px;
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
                margin: 10px;
                width: 192px;
                height: 47px;
                border-radius: 5px;
                background: #DEDEDE;
                color: black;
                border: none;
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
                width: 42px;
                height: 78px;
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
        return os.path.abspath(os.path.join(os.path.dirname(__file__), f'../icon/{image_name}'))

    def remove_file(self):
        # This method removes the widget from its parent
        self.setParent(None)

    def download_file(self):
        # Implement the file download logic here
        print("Download button clicked")

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
