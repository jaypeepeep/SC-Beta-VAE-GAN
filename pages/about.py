"""
Program Title: About Page
Programmer/s:
- Alpapara, Nichole N.
- Lagatuz, John Patrick D.
- Peroche, John Mark P.
- Torreda, Kurt Denver P.
Description: The About page serves as the third page that can be accessed by the user in the system. It provides a
user-friendly interface that contains information about how to use the system, and the thesis paper of the developers.
PyQt5 and os were utilized for the interface and pathing respectively. The setupUi function is responsible for displaying
the user interface while open_pdf_viewer allows its users to open the PDF file of the thesis paper.
Date Added: June 20, 2024
Last Date Modified: November 17, 2024

"""

from PyQt5 import QtWidgets, QtGui, QtCore
import os
from components.widget.pdf_viewer import PDFViewer

class About(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(About, self).__init__(parent)
        self.pdf_viewer = None  # Keep a reference to the PDFViewer instance
        self.setupUi()  # Set up the user interface

    def setupUi(self):
        # Set up the grid layout for the widget
        self.gridLayout = QtWidgets.QGridLayout(self)

        # Create and configure the QTextEdit to display HTML content
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setHtml(
            # HTML content providing information about SC-β-VAE-GAN and steps for usage
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
            "<p><b>Steps in Using this Tool</b></p>"
            "<p>    <b><i>For Workplace*</i></b></p>"
            "<ul>"
            "<li>       <b>Input Files:</b> Input .svc files to augment</li>"
            "<li>       <b>Multiple Inputs:</b> Add more files by clicking 'Add More.'</li>"
            "<li>       <b>Preview Section:</b> Review the data you entered.</li>"
            "<li>       <b>Delete Input:</b> You can delete input by clicking the 'X' button.'</li>"
            "<li>       <b>Train Model:</b> If the model doesn't exist, train the model first with the selected data</li>"
            "<li>       <b>Select Model:</b> Select a pre-trained model to use in generating</li>"
            "<li>       <b>Number of Synthetic Data:</b> Enter the number of data points to generate.</li>"
            "<li>       <b>Generate Synthetic Data:</b> Click to generate synthetic data.</li>"
            "<li>       <b>Results:</b> Check the generated results.</li>"
            "</ul>"
            "<p>    <b><i>      For Handwriting*</i></b></p>"
            "<ul>"
            "<li>       <b>Start Drawing:</b> Click the 'Start Handwriting' button.</li>"
            "<li>       <b>Handwrite or Draw:</b> Begin drawing or writing. If something goes wrong, click 'Clear Drawing.'</li>"
            "<li>       <b>Finish Drawing:</b> When you're done, click 'Done' to return to the main page.</li>"
            "<li>       <b>Multiple Drawings:</b> Add more handwriting by clicking 'Draw More.'</li>"
            "<li>       <b>Clear Drawing:</b> Clear all handwriting by clicking 'Clear All.'</li>"
            "<li>       <b>Number of Synthetic Data:</b> Enter the number of data points to generate.</li>"
            "<li>       <b>Preview Section:</b> Review the data you entered.</li>"
            "<li>       <b>Generate Synthetic Data:</b> Click to generate synthetic data.</li>"
            "<li>       <b>Results:</b> Check the generated results.</li>"
            "</ul>"
        )
        self.textEdit.setReadOnly(True)  # Make the text edit read-only
        self.textEdit.setStyleSheet(
            # Set the stylesheet for the QTextEdit
            "border: none;"
            "background: transparent;"
            "font-size: 13px;"
            "font-family: 'Montserrat', sans-serif;"
            "line-height: 24.38px;"
            "padding: 10px 20px;"
        )
        self.gridLayout.addWidget(self.textEdit, 1, 0, 1, 1)  # Add the QTextEdit to the layout

        # Create and style the button to open the PDF viewer
        self.pdf_button = QtWidgets.QPushButton("View Study", self)
        self.pdf_button.setStyleSheet(
            "QPushButton {"
            "    margin-left: 10px;"
            "    background-color: #003333;"
            "    color: white;"
            "    border: none;"
            "    padding: 5px 15px;"
            "    border-radius: 5px;"
            "    font-size: 10px;"
            "    font-weight: bold;"
            "    font-family: 'Montserrat', sans-serif;"
            "    line-height: 20px;"
            "    margin-bottom: 20px"
            "}"
            "QPushButton:hover {"
            "    background-color: #005555;"
            "}"
        )
        self.pdf_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))  # Set the button cursor
        self.pdf_button.clicked.connect(self.open_pdf_viewer)  # Connect the button click to the PDF viewer function

        # Add the button to the layout
        self.gridLayout.addWidget(self.pdf_button, 2, 0, 1, 1, QtCore.Qt.AlignCenter)

    def open_pdf_viewer(self):
        # Open the PDF viewer with the specified file
        file_name = "main_paper.pdf"  # Name of the PDF file
        file_path = os.path.join(os.path.dirname(__file__), '../paper', file_name)  # Path to the PDF file

        # Check if the file exists
        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.critical(self, "Error", "PDF file not found!")  # Show error if the file is missing
            return

        # Open the PDF in the PDFViewer component
        self.pdf_viewer = PDFViewer(file_path)
        self.pdf_viewer.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)  # Initialize the Qt application
    about = About()  # Create an instance of the About widget
    about.show()  # Show the About widget
    sys.exit(app.exec_())  # Execute the application event loop
