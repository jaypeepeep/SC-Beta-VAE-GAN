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
        self.pdf_viewer = None
        self.setupUi()

    def setupUi(self):
        # Create scroll area for the entire page
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
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

        # Create main content widget
        content_widget = QtWidgets.QWidget()
        content_widget.setStyleSheet("background-color: white;")
        scroll_area.setWidget(content_widget)

        # Main layout
        self.main_layout = QtWidgets.QVBoxLayout(content_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

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
            "<p style='text-align: center;'>View the full study to understand the complete framework, methodology, and key insights behind "
            "<span style='color: #005555; font-weight: bold;'>SC-β-VAE-GAN</span>.</p>"
        )
        intro_text.setWordWrap(True)
        intro_text.setStyleSheet(
            "font-size: 20px;"
            "font-family: 'Montserrat', sans-serif;"
            "line-height: 24.38px;"
        )
        self.main_layout.addWidget(intro_text)

        # PDF Button
        self.pdf_button = QtWidgets.QPushButton("View Study", self)
        self.pdf_button.setStyleSheet(
            "QPushButton {"
            "    background-color: #003333;"
            "    color: white;"
            "    border: none;"
            "    padding: 15px 30px;"
            "    border-radius: 8px;"
            "    font-size: 20px;"
            "    font-weight: bold;"
            "    font-family: 'Montserrat', sans-serif;"
            "}"
            "QPushButton:hover {"
            "    background-color: #005555;"
            "}"
        )
        self.pdf_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pdf_button.clicked.connect(self.open_pdf_viewer)

        # Center the button
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.pdf_button)
        button_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(button_container)

        # Steps section container
        steps_container = QtWidgets.QWidget()
        steps_layout = QtWidgets.QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(20, 20, 20, 20)
        steps_layout.setSpacing(20)

        # Steps title
        steps_title = QtWidgets.QLabel("HOW TO USE THE TOOL")
        steps_title.setAlignment(QtCore.Qt.AlignCenter) 
        steps_title.setStyleSheet(
            "font-size: 25px;"
            "font-family: 'Montserrat', sans-serif;"
            "font-weight: bold;"
            "background-color: transparent;"
        )
        steps_layout.addWidget(steps_title)

        # Create tables container
        tables_container = QtWidgets.QWidget()
        tables_layout = QtWidgets.QHBoxLayout(tables_container)
        tables_layout.setSpacing(20)
        tables_layout.setContentsMargins(0, 0, 0, 0)

        # Create tables for workplace and handwriting
        workplace_table = QtWidgets.QTableWidget()
        handwriting_table = QtWidgets.QTableWidget()

        # Common table style
        table_style = """
            QTableWidget {
                background-color: white;
                border-radius: 8px;
                border: none;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 15px;
                font-weight: bold;
                font-size: 20px;
                font-family: 'Montserrat', sans-serif;
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTableWidget::item {
                padding: 15px;
                font-size: 20px;
                font-family: 'Montserrat', sans-serif;
                line-height: 1.6;
            }
        """
        workplace_table.setStyleSheet(table_style)
        handwriting_table.setStyleSheet(table_style)

        # Configure workplace table
        workplace_steps = [
            "<p>1. <b>Input Files:</b> Input .svc files to augment</p>",
            "<p>2. <b>Multiple Inputs:</b> Add more files by clicking 'Add More.'</p>",
            "<p>3. <b>Preview Section:</b> Review the data you entered.</p>",
            "<p>4. <b>Delete Input:</b> You can delete input by clicking the 'X' button.</p>",
            "<p>5. <b>Train Model:</b> If the model doesn't exist, train the model first with the selected data</p>",
            "<p>6. <b>Select Model:</b> Select a pre-trained model to use in generating</p>",
            "<p>7. <b>Number of Synthetic Data:</b> Enter the number of data points to generate.</p>",
            "<p>8. <b>Generate Synthetic Data:</b> Click to generate synthetic data.</p>",
            "<p>9. <b>Results:</b> Check the generated results.</p>"
        ]
        self.setup_table(workplace_table, "For Workplace", workplace_steps)

        # Configure handwriting table
        handwriting_steps = [
            "<p>1. <b>Start Drawing:</b> Click the 'Start Handwriting' button.</p>",
            "<p>2. <b>Handwrite or Draw:</b> Begin drawing or writing.</p>",
            "<p>3. <b>Finish Drawing:</b> When you're done, click 'Done' to return to the main page.</p>",
            "<p>4. <b>Multiple Drawings:</b> Add more handwriting by clicking 'Draw More.'</p>",
            "<p>5. <b>Clear Drawing:</b> Clear all handwriting by clicking 'Clear All.'</p>",
            "<p>6. <b>Number of Synthetic Data:</b> Enter the number of data points to generate.</p>",
            "<p>7. <b>Preview Section:</b> Review the data you entered.</p>",
            "<p>8. <b>Generate Synthetic Data:</b> Click to generate synthetic data.</p>",
            "<p>9. <b>Results:</b> Check the generated results.</p>"
        ]
        self.setup_table(handwriting_table, "For Handwriting", handwriting_steps)

        # Add tables to layout
        tables_layout.addWidget(workplace_table)
        tables_layout.addWidget(handwriting_table)

        # Add tables container to steps layout
        steps_layout.addWidget(tables_container)

        # Add steps container to main layout
        self.main_layout.addWidget(steps_container)

        # Set up the main layout for the scroll area
        main_scroll_layout = QtWidgets.QVBoxLayout(self)
        main_scroll_layout.setContentsMargins(0, 0, 0, 0)
        main_scroll_layout.addWidget(scroll_area)

    def setup_table(self, table, title, steps):
        # Set table properties
        table.setColumnCount(1)
        table.setRowCount(len(steps))
        table.setHorizontalHeaderLabels([title])
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        
        # Create label for content
        for row, step in enumerate(steps):
            content_label = QtWidgets.QLabel(step)
            content_label.setWordWrap(True)
            content_label.setStyleSheet(
                "font-size: 20px;"
                "font-family: 'Montserrat', sans-serif;"
                "line-height: 10x;"
                "padding: 0px;"
            )
        # Add content to table
            table.setCellWidget(row, 0, content_label)
            table.resizeRowToContents(row)

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