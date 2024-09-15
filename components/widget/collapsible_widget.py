from PyQt5 import QtWidgets, QtCore

class CollapsibleWidget(QtWidgets.QWidget):
    def __init__(self, title="Show More", parent=None):
        super(CollapsibleWidget, self).__init__(parent)
        self.title = title
        self.setupUi()

    def setupUi(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # Toggle button with style and size policies
        self.toggle_button = QtWidgets.QPushButton(self.title, self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        
        # Apply custom styling to the button
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #EBEBEB;
                border-radius: 10px;
                color: #000000;
                font-family: 'Montserrat';
                font-size: 14px;
                font-weight: 600;
                text-align: left;
                padding: 20px;
            }
            QPushButton:checked {
                background-color: #DADADA; /* Slightly darker for checked state */
            }
        """)
        
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
                padding: 10px;  /* Optional: adds padding around content */
            }
        """)

        self.main_layout.addWidget(self.collapsible_container)

        # Connect button toggle signal to show/hide the container
        self.toggle_button.toggled.connect(self.toggle_container)

    def toggle_container(self, checked):
        self.collapsible_container.setVisible(checked)
        if checked:
            self.toggle_button.setText(self.title)
        else:
            self.toggle_button.setText(self.title)

    def add_widget(self, widget):
        """Add a widget to the collapsible area."""
        self.collapsible_layout.addWidget(widget)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = CollapsibleWidget()
    window.show()
    sys.exit(app.exec_())
