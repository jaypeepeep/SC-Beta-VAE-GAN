import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream
from PyQt5.QtGui import QFontDatabase, QFont
from PyQt5.QtGui import QIcon
from layout import Ui_MainWindow



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.setWindowTitle("SC-β-VAE-GAN")
        self.setWindowIcon(QIcon('./icon/icon.ico'))  
        
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

    def on_local_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(3)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load Montserrat fonts
    regular_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-Regular.ttf")
    bold_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-Bold.ttf")
    extrabold_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-ExtraBold.ttf")
    black_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-Black.ttf")
    medium_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-Medium.ttf")
    italic_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-Italic.ttf")
    semibold_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-SemiBold.ttf")
    thin_font_id = QFontDatabase.addApplicationFont("./font/Montserrat-Thin.ttf")

    # Get font family names 
    regular_font_family = QFontDatabase.applicationFontFamilies(regular_font_id)[0]
    bold_font_family = QFontDatabase.applicationFontFamilies(bold_font_id)[0]
    extrabold_font_family = QFontDatabase.applicationFontFamilies(extrabold_font_id)[0]
    black_font_family = QFontDatabase.applicationFontFamilies(black_font_id)[0]
    medium_font_family = QFontDatabase.applicationFontFamilies(medium_font_id)[0]
    italic_font_family = QFontDatabase.applicationFontFamilies(italic_font_id)[0]
    semibold_font_family = QFontDatabase.applicationFontFamilies(semibold_font_id)[0]
    thin_font_family = QFontDatabase.applicationFontFamilies(thin_font_id)[0]

    # Set the default font for the application
    app.setFont(QFont(regular_font_family, 15))  # Set the default font to regular

    # Load and apply the stylesheet
    style_file = QFile("style.qss")
    if style_file.open(QFile.ReadOnly | QFile.Text):
        style_stream = QTextStream(style_file)
        app.setStyleSheet(style_stream.readAll())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())