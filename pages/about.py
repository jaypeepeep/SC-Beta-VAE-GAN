from PyQt5 import QtWidgets, QtGui, QtCore
import webbrowser  # For opening URLs

class About(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(About, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)

        # Text area for the description with HTML formatting
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setHtml(
            "<p>SC-β-VAE-GAN stands for Shift Correction β-Variational Autoencoder-Generative Adversarial Network. It is a hybrid model designed to address the challenges of imputing and augmenting handwriting multivariate time series data.</p>"
            "<p><b>Key Components</b></p>"
            "<ul>"
            "<li><b>Variational Autoencoder (VAE):</b> A type of generative model that learns the underlying distribution of data to generate new, similar data samples. It is particularly useful for capturing the latent space of the data and generating synthetic datasets.</li>"
            "<li><b>Generative Adversarial Network (GAN):</b> A generative model consisting of two neural networks, a generator and a discriminator, that are trained simultaneously. The generator creates fake data samples, while the discriminator attempts to distinguish between real and fake samples. This adversarial process improves the quality of the generated data.</li>"
            "<li><b>Shift Correction (SC):</b> A method incorporated into the model to correct shifts in the data. This is crucial for ensuring the temporal coherence and accuracy of the generated time series data, especially in the context of handwriting where shifts can occur due to various factors like hand movement or writing speed.</li>"
            "</ul>"
            "<p><b>Objectives</b></p>"
            "<ul>"
            "<li><b>Data Imputation:</b> Filling in missing values in multivariate time series data. In handwriting analysis, missing data can occur due to various reasons such as pen lift-offs or sensor errors. The SC-β-VAE-GAN aims to accurately impute these missing values by leveraging the combined strengths of VAEs and GANs.</li>"
            "<li><b>Data Augmentation:</b> Generating additional synthetic data to expand the available dataset. This is particularly useful in handwriting analysis, where collecting large amounts of labeled data can be challenging. Augmented data helps improve the training of machine learning models, leading to better generalization and performance.</li>"
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

        # Button to redirect to a website
        self.button = QtWidgets.QPushButton("View the Main Paper", self)
        self.button.setStyleSheet(""" 
            QPushButton {
                margin-left: 10px;
                background-color: #003333;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 11px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
                line-height: 20px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
        """)
        self.button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button.clicked.connect(self.openWebsite)

        # Add the button to the layout
        self.gridLayout.addWidget(self.button, 2, 0, 1, 1, QtCore.Qt.AlignCenter)

    def openWebsite(self):
        # Create a confirmation dialog
        message_box = QtWidgets.QMessageBox(self)
        message_box.setWindowTitle('Confirm')
        message_box.setText("To read the paper, you'll be redirected to an external browser to access Google Docs. Do you want to proceed?")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        message_box.setDefaultButton(QtWidgets.QMessageBox.Yes)

        # Apply stylesheet to the message box buttons
        message_box.setStyleSheet("QPushButton { font-size: 14px; }")

        reply = message_box.exec_()  # Display the dialog and wait for user response

        if reply == QtWidgets.QMessageBox.Yes:
            url = "https://docs.google.com/document/d/1sqG_w8LSiaO8grmNdqV8ltWWL9LJOPmJaPLiJ3d9gbU/edit?usp=sharing"  # Replace with the actual URL
            webbrowser.open(url)
