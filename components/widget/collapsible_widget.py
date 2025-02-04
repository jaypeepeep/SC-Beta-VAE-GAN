from PyQt5 import QtWidgets, QtCore, QtGui
import os
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

class CollapsibleWidget(QtWidgets.QWidget):
    def __init__(self, title="Show More", parent=None):
        super(CollapsibleWidget, self).__init__(parent)
        self.title = title
        self.setupUi()

    def setupUi(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        
        # Icon paths
        self.arrow_down_icon_path = self.get_image_path('arrow_down.png')
        self.arrow_up_icon_path = self.get_image_path('arrow_up.png')

        # Icons on the right side
        self.arrow_down_icon = QtGui.QIcon(self.arrow_down_icon_path)
        self.arrow_up_icon = QtGui.QIcon(self.arrow_up_icon_path)

        # Toggle button with style and size policies
        self.toggle_button = QtWidgets.QPushButton(self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        
        # Layout for the button content
        self.button_layout = QtWidgets.QHBoxLayout(self.toggle_button)
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.button_layout.setSpacing(0)  # Remove spacing

        # Label for the text with custom font settings
        self.button_text = QtWidgets.QLabel(self.title, self)
        self.button_text.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.button_text.setStyleSheet("""
            font-family: 'Montserrat';
            font-size: 14px;
            font-weight: 600;
            color: #000000;
            padding: 15px;
        """)

        # Label for the icon
        self.button_icon = QtWidgets.QLabel(self)
        self.button_icon.setPixmap(self.arrow_down_icon.pixmap(15, 15))
        self.button_icon.setStyleSheet("padding-right: 15px;")

        # Text and icon on the button layout
        self.button_layout.addWidget(self.button_text)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.button_icon)
        
        # Custom styling for the button
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #EBEBEB;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:checked {
                background-color: #DADADA; /* Slightly darker for checked state */
            }
            QPushButton:hover {
                background-color: #DADADA; /* Slightly darker for checked state */
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 25))
        self.toggle_button.setGraphicsEffect(shadow)

        self.toggle_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.main_layout.addWidget(self.toggle_button)

        # Collapsible content container with custom style and size policies
        self.collapsible_container = QtWidgets.QWidget(self)
        self.collapsible_container.setVisible(False)
        self.collapsible_layout = QtWidgets.QVBoxLayout(self.collapsible_container)

        # Set size policies to adapt to content
        self.collapsible_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Apply custom styling to the collapsible container
        self.collapsible_container.setStyleSheet("""
            QWidget {
                background-color: #EBEBEB;
                border-radius: 10px;
                padding: 15px;  /* Optional: adds padding around content */
            }
        """)

        container_shadow = QGraphicsDropShadowEffect()
        container_shadow.setBlurRadius(15)
        container_shadow.setXOffset(0)
        container_shadow.setYOffset(2)
        container_shadow.setColor(QtGui.QColor(0, 0, 0, 25))
        self.collapsible_container.setGraphicsEffect(container_shadow)

        self.main_layout.addWidget(self.collapsible_container)

        # Connect button toggle signal to show/hide the container
        self.toggle_button.toggled.connect(self.toggle_container)

    def toggle_container(self, checked):
        self.collapsible_container.setVisible(checked)
        if checked:
            self.button_text.setText(self.title)
            self.button_icon.setPixmap(self.arrow_up_icon.pixmap(15, 15))
        else:
            self.button_text.setText(self.title)
            self.button_icon.setPixmap(self.arrow_down_icon.pixmap(15, 15))

    def add_widget(self, widget):
        """Add a widget to the collapsible area."""
        self.collapsible_layout.addWidget(widget)

    def get_image_path(self, image_name):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../icon/{image_name}'))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = CollapsibleWidget()
    window.show()
    sys.exit(app.exec_())
