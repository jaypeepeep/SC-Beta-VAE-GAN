import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream
from components.full_menu_widget import FullMenuWidget

from sidebar_sample import Ui_MainWindow
from components.full_menu_widget import FullMenuWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load and apply the stylesheet
    style_file = QFile("style.qss")
    if style_file.open(QFile.ReadOnly | QFile.Text):
        style_stream = QTextStream(style_file)
        app.setStyleSheet(style_stream.readAll())
        style_file.close()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
