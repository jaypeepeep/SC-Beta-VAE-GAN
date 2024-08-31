from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        # Create a QGridLayout
        self.gridLayout = QtWidgets.QGridLayout(self)
        
        # Create QWebEngineView and load HTML file
        self.web_view = QWebEngineView(self)
        self.gridLayout.addWidget(self.web_view, 0, 0)  # Add QWebEngineView to the layout
        
        # Load the HTML file (make sure the path is correct)
        self.web_view.setUrl(QtCore.QUrl.fromLocalFile(r'C:\Users\patri\Thesis-Project\components\canvas\Canvas.html'))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.show()
    sys.exit(app.exec_())
