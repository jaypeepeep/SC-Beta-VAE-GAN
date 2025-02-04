from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QIcon
import fitz  # PyMuPDF
import os
import shutil  # To copy the file to a user-selected location

class PDFViewer(QtWidgets.QWidget):
    def __init__(self, pdf_path, parent=None):
        super(PDFViewer, self).__init__(parent)
        self.pdf_path = pdf_path
        self.zoom_factor = 1.0  # Initial zoom level
        self.setupUi()
        self.load_pdf()

    def setupUi(self):
        self.setWindowTitle("SC-β-VAE-GAN: A Shift Correction VAE-GAN Model for Imputation and Augmentation of Handwriting Multivariate Time Series Data")
        self.setWindowIcon(QIcon(os.path.abspath('icon/icon.ico')))

        # Set window size to a fraction of the screen
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setGeometry(screen.width() // 4, screen.height() // 4, screen.width() // 2, screen.height() // 2)

        self.gridLayout = QtWidgets.QGridLayout(self)

        # Scroll Area for the PDF content
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.gridLayout.addWidget(self.scroll_area, 0, 0)

        # Widget to contain all PDF pages
        self.pages_container = QtWidgets.QWidget()
        self.pages_layout = QtWidgets.QVBoxLayout(self.pages_container)
        self.pages_layout.setContentsMargins(0, 0, 0, 0)
        self.pages_layout.setSpacing(10)  # Space between pages

        self.scroll_area.setWidget(self.pages_container)

        # Zoom Slider
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.zoom_slider.setRange(10, 300)  # Zoom range from 10% to 300%
        self.zoom_slider.setValue(100)  # Default to 100%
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.gridLayout.addWidget(self.zoom_slider, 1, 0)

        # Download Button
        self.download_button = QtWidgets.QPushButton("Download PDF", self)
        self.download_button.clicked.connect(self.on_download_button_clicked)
        self.gridLayout.addWidget(self.download_button, 2, 0)

    def load_pdf(self):
        if not os.path.exists(self.pdf_path):
            QtWidgets.QMessageBox.critical(self, "Error", "PDF file not found!")
            return

        try:
            self.doc = fitz.open(self.pdf_path)
            self.display_all_pages()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load PDF: {e}")

    def display_all_pages(self):
        # Clear the previous pages layout
        for i in reversed(range(self.pages_layout.count())):
            widget = self.pages_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for page_number in range(self.doc.page_count):
            try:
                page = self.doc.load_page(page_number)
                mat = fitz.Matrix(self.zoom_factor, self.zoom_factor)
                pix = page.get_pixmap(matrix=mat)
                image = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, QtGui.QImage.Format_RGB888)

                # Create a label for the image
                image_label = QtWidgets.QLabel(self)
                image_label.setPixmap(QtGui.QPixmap.fromImage(image))
                image_label.setAlignment(QtCore.Qt.AlignCenter)

                # Create a horizontal layout for centering the image
                image_layout = QtWidgets.QHBoxLayout()
                image_layout.addStretch(1)  # Add stretchable space on the left
                image_layout.addWidget(image_label)  # Center the image label
                image_layout.addStretch(1)  # Add stretchable space on the right

                # Create a container widget for the image layout
                image_container = QtWidgets.QWidget(self)
                image_container.setLayout(image_layout)

                # Add the image container to the main layout
                self.pages_layout.addWidget(image_container)

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to display page {page_number + 1}: {e}")

    def on_zoom_changed(self, value):
        self.zoom_factor = value / 100.0  # Convert slider value to zoom factor
        self.display_all_pages()  # Re-render pages with the new zoom factor

    def on_download_button_clicked(self):
        default_filename = "SC-β-VAE-GAN A SHIFT CORRECTION VAE-GAN MODEL FOR IMPUTATION AND AUGMENTATION OF HANDWRITING MULTIVARIATE TIME SERIES DATA.pdf"
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PDF", default_filename, "PDF Files (*.pdf)")

        if save_path:
            try:
                # Copy the PDF file to the selected path
                shutil.copy(self.pdf_path, save_path)
                QtWidgets.QMessageBox.information(self, "Download Successful", "PDF downloaded successfully!")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to download PDF: {e}")

    def resizeEvent(self, event):
        # Center the PDF in the scroll area when the window is resized
        self.scroll_area.horizontalScrollBar().setValue((self.scroll_area.widget().width() - self.scroll_area.width()) // 2)
        self.scroll_area.verticalScrollBar().setValue((self.scroll_area.widget().height() - self.scroll_area.height()) // 2)
        super(PDFViewer, self).resizeEvent(event)
