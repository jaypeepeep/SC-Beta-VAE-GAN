from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QMessageBox
from PyQt5.QtGui import QColor
import os
import zipfile
import shutil
import tempfile

class OutputWidget(QtWidgets.QWidget):
    clearUI = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super(OutputWidget, self).__init__(parent)
        self.parent = parent
        self.output_zip_path = None
        self.setup_output_collapsible()
        
    def setup_output_collapsible(self):
        """Set up the 'Output' collapsible widget"""
        layout = QtWidgets.QVBoxLayout(self)

        self.file_container = QtWidgets.QWidget(self)
        self.file_container.setStyleSheet("""
            background-color: #DEDEDE;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 0;
        """)

        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(8)
        shadow_effect.setColor(QColor(0, 0, 0, 160))
        shadow_effect.setOffset(2)
        self.file_container.setGraphicsEffect(shadow_effect)

        file_container_layout = QtWidgets.QHBoxLayout(self.file_container)
        file_container_layout.setSpacing(20)
        file_container_layout.setContentsMargins(10, 10, 10, 10)

        self.file_label = QtWidgets.QLabel("Time-series_Data.zip")
        self.file_label.setStyleSheet("font-family: Montserrat; font-size: 14px; color: black;")
        file_container_layout.addWidget(self.file_label)

        button_container = QtWidgets.QWidget(self)
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setSpacing(10)

        done_button = QtWidgets.QPushButton("Done")
        done_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #003333;
                color: #003333;
                background-color: transparent;
                font-family: Montserrat;
                font-weight: 600;
                font-size: 14px;
                padding: 7px 20px;
                border-radius: 5px;
            }QPushButton:hover {
                background-color: #005555;
                color: white;
            }
        """)
        done_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        done_button.clicked.connect(self.handle_done_click)
        button_layout.addWidget(done_button)

        download_button = QtWidgets.QPushButton("Download")
        download_button.setStyleSheet("""
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 14px; 
                font-weight: 600; 
                padding: 8px 16px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
        """)
        download_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        download_button.clicked.connect(self.handle_download_click)
        button_layout.addWidget(download_button)

        delete_button = QtWidgets.QPushButton("X")
        delete_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                font-weight: 600;
                color: #FF5252;
                font-family: Montserrat;
                font-size: 22px;
                padding: 5px;
            }
        """)
        delete_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        delete_button.clicked.connect(self.handle_remove_click)
        button_layout.addWidget(delete_button)

        file_container_layout.addWidget(button_container, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.file_container)

    def handle_done_click(self):
        reply = QMessageBox.question(
            self,
            'Confirmation',
            "Are you done generating synthetic data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            finish_msg = QMessageBox.information(
                self,
                'Finished',
                "Finished Generating Synthetic Data",
                QMessageBox.Ok
            )
            self.clearUI.emit()  # Emit signal to clear the UI

    def handle_download_click(self):
        if not self.output_zip_path:
            self.create_output_zip()
        
        if self.output_zip_path and os.path.exists(self.output_zip_path):
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Zip File",
                "Time-series_Data.zip",
                "Zip files (*.zip)"
            )
            if save_path:
                try:
                    shutil.copy2(self.output_zip_path, save_path)
                    QMessageBox.information(
                        self,
                        'Success',
                        f"File saved successfully to {save_path}",
                        QMessageBox.Ok
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        'Error',
                        f"Error saving file: {str(e)}",
                        QMessageBox.Ok
                    )

    def handle_remove_click(self):
        reply = QMessageBox.question(
            self,
            'Confirmation',
            "Are you sure you want to remove the output file?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.output_zip_path and os.path.exists(self.output_zip_path):
                try:
                    os.remove(self.output_zip_path)
                    self.output_zip_path = None
                    QMessageBox.information(
                        self,
                        'Success',
                        "Output file has been removed",
                        QMessageBox.Ok
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        'Error',
                        f"Error removing file: {str(e)}",
                        QMessageBox.Ok
                    )

    def create_output_zip(self):
        try:
            # Get the input files from the parent Workplace widget
            input_files = self.parent.uploaded_files if hasattr(self.parent, 'uploaded_files') else []
            
            if not input_files:
                QMessageBox.warning(
                    self,
                    'No Input Files',
                    "No input files found to create zip file",
                    QMessageBox.Ok
                )
                return

            # Create a temporary zip file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                self.output_zip_path = temp_zip.name

            # Create the zip file with input files
            with zipfile.ZipFile(self.output_zip_path, 'w') as zipf:
                for file_path in input_files:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))

        except Exception as e:
            QMessageBox.warning(
                self,
                'Error',
                f"Error creating zip file: {str(e)}",
                QMessageBox.Ok
            )
            self.output_zip_path = None