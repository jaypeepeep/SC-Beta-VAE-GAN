from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QMessageBox
from PyQt5.QtGui import QColor
import os
import shutil

class OutputWidget(QtWidgets.QWidget):
    clearUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(OutputWidget, self).__init__(parent)
        self.parent = parent
        self.output_zip_path = None
        self.setup_output_collapsible()
        self.setVisible(False) 

    def setup_output_collapsible(self):
        """Set up the 'Output' collapsible widget"""
        layout = QtWidgets.QVBoxLayout(self)

        self.file_container = QtWidgets.QWidget(self)
        self.file_container.setStyleSheet("""
            background-color: #DEDEDE;
            padding: 5px;
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
                font-size: 10px;
                padding: 5px 15px;
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
                font-size: 10px; 
                font-weight: 600; 
                padding: 5px 15px; 
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
                font-size: 20px;
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
            finish_msg.setStyleSheet("""
                QMessageBox {
                    font-size: 12px;
                    font-weight: bold;
                    margin: 32px 32px;
                    
                    font-family: 'Montserrat', sans-serif;
                }
                QPushButton {
                    margin-left: 10px;
                    background-color: #003333;
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 5px;
                    font-size: 10px;
                    font-weight: bold;
                    font-family: 'Montserrat', sans-serif;
                    line-height: 20px;
                }
                QPushButton:hover {
                    background-color: #005555;
                }
            """)
            self.clearUI.emit()

    def handle_download_click(self):
        if self.output_zip_path and os.path.exists(self.output_zip_path):
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Zip File",
                "Time-series_Data.zip",
                "Zip files (*.zip)"
            )
            if save_path:
                try:
                    # Try to copy the file
                    shutil.copy2(self.output_zip_path, save_path)
                    
                    # Create and style the success message box
                    success_box = QtWidgets.QMessageBox(self)
                    success_box.setIcon(QtWidgets.QMessageBox.Information)
                    success_box.setWindowTitle('Success')
                    success_box.setText(f"File saved successfully to {save_path}")
                    success_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    
                    # Apply custom styles for success box
                    success_box.setStyleSheet("""
                        QMessageBox {
                            font-size: 12px;
                            font-weight: bold;
                            margin: 32px 32px;
                            font-family: 'Montserrat', sans-serif;
                            color: #333; /* Text color */
                        }
                        QPushButton {
                            margin-left: 10px;
                            background-color: #003333;
                            color: white;
                            border: none;
                            padding: 5px 15px;
                            border-radius: 5px;
                            font-size: 10px;
                            font-weight: bold;
                            font-family: 'Montserrat', sans-serif;
                        }
                        QPushButton:hover {
                            background-color: #005555;
                        }
                        QPushButton:pressed {
                            background-color: #002222;
                        }
                    """)

                    # Show success message box
                    success_box.exec_()

                except Exception as e:
                    # Create and style the error message box
                    error_box = QtWidgets.QMessageBox(self)
                    error_box.setIcon(QtWidgets.QMessageBox.Warning)
                    error_box.setWindowTitle('Error')
                    error_box.setText(f"Error saving file: {str(e)}")
                    error_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    
                    # Apply custom styles for error box
                    error_box.setStyleSheet("""
                        QMessageBox {
                            font-size: 12px;
                            font-weight: bold;
                            margin: 32px 32px;
                            font-family: 'Montserrat', sans-serif;
                            color: #333; /* Text color */
                        }
                        QPushButton {
                            margin-left: 10px;
                            background-color: #003333;
                            color: white;
                            border: none;
                            padding: 5px 15px;
                            border-radius: 5px;
                            font-size: 10px;
                            font-weight: bold;
                            font-family: 'Montserrat', sans-serif;
                        }
                        QPushButton:hover {
                            background-color: #005555;
                        }
                        QPushButton:pressed {
                            background-color: #002222;
                        }
                    """)

                    # Show error message box
                    error_box.exec_()

    def handle_remove_click(self):
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle('Confirmation')
        message_box.setText("Are you sure you want to remove the output file?")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        message_box.setDefaultButton(QtWidgets.QMessageBox.No)

        # Apply custom stylesheet to the message box
        message_box.setStyleSheet("""
            QMessageBox {
                font-size: 12px;
                font-weight: bold;
                margin: 32px 32px;
                font-family: 'Montserrat', sans-serif;
                color: #333;
            }
            QPushButton {
                margin-left: 10px;
                background-color: #003333;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 5px;
                font-size: 10px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
                line-height: 20px;
            }
            QPushButton:hover {
                background-color: #005555;
            }
        """)

        # Execute the message box and capture the user's response
        reply = message_box.exec_()

        # Handle the response
        if reply == QtWidgets.QMessageBox.Yes:
            # Proceed with removing the output file
            print("Removing the output file...")
        else:
            # Do nothing if 'No' is selected
            print("Canceling the removal.")
        
        if reply == QMessageBox.Yes:
            if self.output_zip_path and os.path.exists(self.output_zip_path):
                try:
                    os.remove(self.output_zip_path)
                    self.output_zip_path = None
                    self.setVisible(False)
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

    @QtCore.pyqtSlot(str)
    def set_zip_path(self, zip_path):
        """Set the ZIP file path for the OutputWidget."""
        try:
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")

            self.output_zip_path = zip_path
            self.file_label.setText(os.path.basename(zip_path))  # Update the label with the zip filename
            self.setVisible(True)  # Ensure the widget is visible
        except FileNotFoundError as e:
            self.show_error_message(f"File error: {str(e)}")
        except Exception as e:
            self.show_error_message(f"Unexpected error: {str(e)}")

    def show_error_message(self, message):
        """Show a message box with the error message."""
        # Create the error message box
        error_box = QtWidgets.QMessageBox(self)
        error_box.setIcon(QtWidgets.QMessageBox.Warning)
        error_box.setWindowTitle('Error')
        error_box.setText(message)
        error_box.setStandardButtons(QtWidgets.QMessageBox.Ok)

        # Apply custom stylesheet to the error message box
        error_box.setStyleSheet("""
            QMessageBox {
                font-size: 12px;
                font-weight: bold;
                margin: 32px 32px;
                font-family: 'Montserrat', sans-serif;
                color: #333;
            }
            QPushButton {
                margin-left: 10px;
                background-color: #003333;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 5px;
                font-size: 10px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
            }
            QPushButton:hover {
                background-color: #005555;
            }
            QPushButton:pressed {
                background-color: #002222;
            }
        """)

        # Show the error message box
        error_box.exec_()