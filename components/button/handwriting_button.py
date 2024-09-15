import os
from PyQt5 import QtWidgets, QtCore, QtGui

class handwritingButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super(handwritingButton, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # Set a more reasonable size for the button
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        self.setStyleSheet(
                """
            QPushButton {
                background-color: white;
                color: #033;
                border: 2px solid #033;
                border-radius: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #A3BFBF;  
                border-color: #055;         
                color: #055;                
            }
            """
        )

        # Create a layout for the button content (icon and text)
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setAlignment(QtCore.Qt.AlignCenter)  # Center items horizontally in the button

        # Add icon to QPushButton
        icon = QtWidgets.QLabel(self)
        icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../icon/handwriting icon.png'))
        
        if os.path.exists(icon_path):
            pixmap = QtGui.QPixmap(icon_path)
            pixmap = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            icon.setPixmap(pixmap)
        else:
            icon.setText("Image not found")
        
        icon.setAlignment(QtCore.Qt.AlignCenter)
        icon.setStyleSheet("border: none;")

        # Add text to QPushButton
        button_text = QtWidgets.QLabel("Click to Start", self)
        button_text.setAlignment(QtCore.Qt.AlignCenter)
        button_text.setStyleSheet("font-size: 25px; font-weight: 500; color: #033; border: none;")

        # Add icon and text to the button layout
        button_layout.addWidget(icon)
        button_layout.addWidget(button_text)
        button_layout.setContentsMargins(0,0,0,0)  # Add margins around the content
        button_layout.setSpacing(10)  # Space between icon and text

        # Set the layout for the QPushButton
        self.setLayout(button_layout)
