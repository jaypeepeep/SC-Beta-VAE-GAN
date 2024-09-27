from PyQt5 import QtWidgets, QtGui, QtCore
import os
from components.widget.pdf_viewer import PDFViewer

class About(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(About, self).__init__(parent)
        self.pdf_viewer = None  # Keep a reference to the PDFViewer instance
        self.setupUi()

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)

        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setHtml(
            "<p>SC-β-VAE-GAN stands for Shift Correction β-Variational Autoencoder-Generative Adversarial Network. "
            "It is a hybrid model designed to address the challenges of imputing and augmenting handwriting multivariate time series data.</p>"
            "<p><b>Key Components</b></p>"
            "<ul>"
            "<li><b>Variational Autoencoder (VAE):</b> A type of generative model that learns the underlying distribution of data to generate new, similar data samples.</li>"
            "<li><b>Generative Adversarial Network (GAN):</b> A generative model consisting of two neural networks that are trained simultaneously.</li>"
            "<li><b>Shift Correction (SC):</b> A method incorporated into the model to correct shifts in the data.</li>"
            "</ul>"
            "<p><b>Objectives</b></p>"
            "<ul>"
            "<li><b>Data Imputation:</b> Filling in missing values in multivariate time series data.</li>"
            "<li><b>Data Augmentation:</b> Generating additional synthetic data to expand the available dataset.</li>"
            "</ul>"
        )
        self.textEdit.setReadOnly(True)
        self.textEdit.setStyleSheet(
            "border: none;"
            "background: transparent;"
            "font-size: 13px;"
            "font-family: 'Montserrat', sans-serif;"
            "line-height: 24.38px;"
            "padding: 10px 20px;"
        )
        self.gridLayout.addWidget(self.textEdit, 1, 0, 1, 1)

        # Button to view PDF in a new window
        self.pdf_button = QtWidgets.QPushButton("View Study", self)
        self.pdf_button.setStyleSheet(
            "QPushButton {"
            "    margin-left: 10px;"
            "    background-color: #003333;"
            "    color: white;"
            "    border: none;"
            "    padding: 10px 20px;"
            "    border-radius: 5px;"
            "    font-size: 11px;"
            "    font-weight: bold;"
            "    font-family: 'Montserrat', sans-serif;"
            "    line-height: 20px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #005555;"
            "}"
        )
        self.pdf_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pdf_button.clicked.connect(self.open_pdf_viewer)

        # Add the button to the layout
        self.gridLayout.addWidget(self.pdf_button, 2, 0, 1, 1, QtCore.Qt.AlignCenter)

    def open_pdf_viewer(self):
        file_name = "main_paper.pdf"
        file_path = os.path.join(os.path.dirname(__file__), '../paper', file_name)

        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.critical(self, "Error", "PDF file not found!")
            return

        self.pdf_viewer = PDFViewer(file_path)
        self.pdf_viewer.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    about = About()
    about.show()
    sys.exit(app.exec_())
