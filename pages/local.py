# local.py

from PyQt5 import QtWidgets, QtGui, QtCore
from components.collapsible_widget import CollapsibleWidget
from components.file_container_widget import FileContainerWidget 

class Local(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Local, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)

        # Main label
        self.label_4 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setText("Local Page")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        self.gridLayout.addWidget(self.collapsible_widget, 1, 0, 1, 1)

        # Add a file container widget to the collapsible widget
        self.file_container = FileContainerWidget("example_file.txt", self)
        self.collapsible_widget.add_widget(self.file_container)
        
        # Ensure only the remove button is visible
        self.file_container.hide_download_button()
        self.file_container.hide_retry_button()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    local_widget = Local()
    local_widget.show()
    sys.exit(app.exec_())
