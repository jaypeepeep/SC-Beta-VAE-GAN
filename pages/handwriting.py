from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineProfile, QWebEngineDownloadItem
from PyQt5.QtCore import QUrl, QStandardPaths
from PyQt5.QtGui import QIcon
import sys

class MyWebEnginePage(QWebEnginePage):
    def __init__(self, parent=None):
        super(MyWebEnginePage, self).__init__(parent)

class MyWebEngineView(QWebEngineView):
    def __init__(self, parent=None):
        super(MyWebEngineView, self).__init__(parent)
        self.page().profile().downloadRequested.connect(self.handleDownloadRequested)

    def handleDownloadRequested(self, download):
        file_name = download.downloadFileName()
        save_path = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation) + '/' + file_name
        download.setPath(save_path)
        download.accept()

        # Show a popup message when the download is complete
        download.finished.connect(lambda: self.showDownloadCompleteMessage(save_path))

    def showDownloadCompleteMessage(self, file_path):
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Information)
        message_box.setWindowTitle("Download Complete")
        message_box.setText(f"File has been downloaded successfully:\n{file_path}")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        message_box.exec_()

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        # Create a QGridLayout
        self.gridLayout = QtWidgets.QGridLayout(self)
        
        # Create QWebEngineView and load HTML file
        self.web_view = MyWebEngineView(self)
        self.gridLayout.addWidget(self.web_view, 0, 0)  # Add QWebEngineView to the layout
        
        # Load the HTML file (make sure the path is correct)
        self.web_view.setUrl(QtCore.QUrl("file:/../components/canvas/Canvas.html"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.show()
    sys.exit(app.exec_())
