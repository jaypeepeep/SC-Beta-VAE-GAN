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
# import this for fonts
from font.dynamic_font_size import get_font_sizes, apply_fonts   
from PyQt5.QtGui import QFont


class ScrollableTableWidget(QtWidgets.QWidget):
    def __init__(self, title, steps, parent=None):
        super(ScrollableTableWidget, self).__init__(parent)
        
        # define first
        font_sizes = get_font_sizes()  
        font_family = "Montserrat"
        titlefont = QtGui.QFont("Montserrat", font_sizes["title"])
        
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QtWidgets.QLabel(title)
        header.setStyleSheet("""
            background-color: #e0e0e0;
            padding: 10px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        """)
        
        # QLABEL - component.setFont()
        header.setFont(titlefont)
        header.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(header)

        # Scrollable content area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Content widget
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Steps
        for step in steps:
            step_label = QtWidgets.QLabel(step)
            step_label.setWordWrap(True)
            step_label.setStyleSheet("""
                font-family: 'Montserrat', sans-serif;
                line-height: 1.2;
                padding: 5px 10px;
                background-color: white;
            """)
            content_layout.addWidget(step_label)

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

class About(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(About, self).__init__(parent)
        self.pdf_viewer = None
        self.setupUi()

    def setupUi(self):
        # Main layout
        font_sizes = get_font_sizes()  
        font_family = "Montserrat"
        contentfont = QtGui.QFont("Montserrat", font_sizes["content"])
        buttonfont = QtGui.QFont("Montserrat", font_sizes["button"])
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Fixed header container
        header_container = QtWidgets.QWidget()
        header_layout = QtWidgets.QVBoxLayout(header_container)
        header_layout.setContentsMargins(20, 20, 20, 20)

        # Introduction text
        intro_text = QtWidgets.QLabel(
            "<p><span style='color: #005555; font-weight: bold;'>SC-β-VAE-GAN</span> stands for Shift Correction β-Variational Autoencoder-Generative Adversarial Network. "
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
            "<p style='text-align: center; '>View the full study to understand the complete framework, methodology, and key insights behind "
            "<span style='color: #005555; font-weight: bold;'>SC-β-VAE-GAN</span>.</p>"
        )
        intro_text.setWordWrap(True)
        # Set only line-height in the stylesheet and rely on setFont() for the font
        intro_text.setStyleSheet(
            "line-height: 24.38px;"  # Only style line-height, not the font
        )

        intro_text.setFont(contentfont)
        header_layout.addWidget(intro_text)

        # PDF Button
        self.pdf_button = QtWidgets.QPushButton("View Study", self)
        self.pdf_button.setStyleSheet(
            "QPushButton {"
            "    background-color: #003333;"
            "    color: white;"
            "    border: none;"
            "    padding: 10px 20px;"
            "    border-radius: 8px;"
            "    font-weight: bold;"
            "    margin-top: 12px;"      
            "}"
            "QPushButton:hover {"
            "    background-color: #005555;"
            "}"
        )
        
        self.pdf_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pdf_button.setFont(buttonfont)
        self.pdf_button.clicked.connect(self.open_pdf_viewer)

        # Center the button
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.pdf_button)
        button_layout.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(button_container)

        # Add header container to main layout
        main_layout.addWidget(header_container)

        # Steps title
        steps_title = QtWidgets.QLabel("HOW TO USE THE TOOL")
        steps_title.setAlignment(QtCore.Qt.AlignCenter)
        steps_title.setStyleSheet(
            "background-color: transparent;"
            "margin: 20px 0;"
            "font-weight: bold"
        )
        steps_title.setFont(contentfont)
        main_layout.addWidget(steps_title)

        # Tables container
        tables_container = QtWidgets.QWidget()
        tables_layout = QtWidgets.QHBoxLayout(tables_container)
        tables_layout.setSpacing(20)
        tables_layout.setContentsMargins(20, 0, 20, 20)

        # Workplace steps
        workplace_steps = [
            "1. <b>Input Files:</b> Input .svc files to augment",
            "2. <b>Multiple Inputs:</b> Add more files by clicking 'Add More.'",
            "3. <b>Preview Section:</b> Review the data you entered.",
            "4. <b>Delete Input:</b> You can delete input by clicking the 'X' button.",
            "5. <b>Train Model:</b> If the model doesn't exist, train the model first with the selected data",
            "6. <b>Select Model:</b> Select a pre-trained model to use in generating",
            "7. <b>Number of Synthetic Data:</b> Enter the number of data points to generate.",
            "8. <b>Generate Synthetic Data:</b> Click to generate synthetic data.",
            "9. <b>Results:</b> Check the generated results."
        ]

        # Handwriting steps
        handwriting_steps = [
            "1. <b>Start Drawing:</b> Click the 'Start Handwriting' button.",
            "2. <b>Handwrite or Draw:</b> Begin drawing or writing.",
            "3. <b>Finish Drawing:</b> When you're done, click 'Done' to return to the main page.",
            "4. <b>Multiple Drawings:</b> Add more handwriting by clicking 'Draw More.'",
            "5. <b>Clear Drawing:</b> Clear all handwriting by clicking 'Clear All.'",
            "6. <b>Number of Synthetic Data:</b> Enter the number of data points to generate.",
            "7. <b>Preview Section:</b> Review the data you entered.",
            "8. <b>Generate Synthetic Data:</b> Click to generate synthetic data.",
            "9. <b>Results:</b> Check the generated results."
        ]
        

        # Create scrollable tables
        workplace_table = ScrollableTableWidget("For Workplace", workplace_steps)
        handwriting_table = ScrollableTableWidget("For Handwriting", handwriting_steps)

        # Add tables to container
        workplace_table.setFont(contentfont) 
        handwriting_table.setFont(contentfont)  
        tables_layout.addWidget(workplace_table)
        tables_layout.addWidget(handwriting_table)
        
        main_layout.addWidget(tables_container)

    def open_pdf_viewer(self):
        file_name = "final_paper.pdf"
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