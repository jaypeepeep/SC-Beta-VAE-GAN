import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream

from sidebar_sample import Ui_MainWindow
from components.full_menu_widget import FullMenuWidget
from components.icon_only_widget import IconOnlyWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.icon_only_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.full_menu_widget.workplace_btn_2.setChecked(True)
    
    ## Change QPushButton Checkable status when stackedWidget index changed
    def on_stackedWidget_currentChanged(self, index):
        btn_list = self.ui.icon_only_widget.findChildren(QPushButton) \
                    + self.ui.full_menu_widget.findChildren(QPushButton)
        
        for btn in btn_list:
            if index in [5, 6]:
                btn.setAutoExclusive(False)
                btn.setChecked(False)
            else:
                btn.setAutoExclusive(True)
    
    ## functions for changing menu page
    def on_workplace_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    
    def on_workplace_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def on_handwriting_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def on_handwriting_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def on_about_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_about_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_local_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_local_btn_2_toggled(self, ):
        self.ui.stackedWidget.setCurrentIndex(3)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load and apply the stylesheet
    style_file = QFile("style.qss")
    if style_file.open(QFile.ReadOnly | QFile.Text):
        style_stream = QTextStream(style_file)
        app.setStyleSheet(style_stream.readAll())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
