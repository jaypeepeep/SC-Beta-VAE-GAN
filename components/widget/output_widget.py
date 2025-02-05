from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QMessageBox
from PyQt5.QtGui import QColor
import os
import shutil
from font.dynamic_font_size import get_font_sizes, apply_fonts
from PyQt5.QtGui import QFont

class OutputWidget(QtWidgets.QWidget):
    clearUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(OutputWidget, self).__init__(parent)
        self.parent = parent
        self.output_zip_path = None
        self.setup_output_collapsible()
        self.setVisible(False)
        font_sizes = get_font_sizes()
        font_family = "Montserrat"
        content_font = QFont(font_family, font_sizes["content"]) 

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
        self.file_label.setStyleSheet("font-family: Montserrat; font-size: {font_sizes['title']}px; color: black;")
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
                font-size: {font_sizes['button']}px;
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
                font-size: {font_sizes['button']}px;
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
                font-size: {font_sizes['button']}px;
                padding: 5px;
            }
        """)
        delete_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        delete_button.clicked.connect(self.handle_remove_click)
        button_layout.addWidget(delete_button)

        file_container_layout.addWidget(button_container, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.file_container)


    def handle_done_click(self):
        # Create the confirmation QMessageBox manually
        confirm_msg = QtWidgets.QMessageBox(self)
        confirm_msg.setWindowTitle("Confirmation")
        confirm_msg.setText("Are you done generating synthetic data?")
        confirm_msg.setIcon(QtWidgets.QMessageBox.Question)

        # Add 'Yes' and 'No' buttons
        yes_button = confirm_msg.addButton(QtWidgets.QMessageBox.Yes)
        no_button = confirm_msg.addButton(QtWidgets.QMessageBox.No)
        confirm_msg.setDefaultButton(no_button)
        # Apply custom styles directly to the buttons
        yes_button.setStyleSheet("""
            background-color: #003333;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 5px;
            font-size: {font_sizes['button']}px;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
        """)

        no_button.setStyleSheet("""
            background-color: #003333;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 5px;
            font-size: {font_sizes['button']}px;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
        """)
        # Apply the stylesheet
        confirm_msg.setStyleSheet("""
            QMessageBox {
                font-size: {font_sizes['content']}px;
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
                font-size: {font_sizes['button']}px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
                line-height: 20px;
            }
            QPushButton:hover {
                background-color: #005555;
            }
            QPushButton:pressed {
                background-color: #002222;
            }
        """)

        # Show the confirmation message box
        confirm_msg.exec_()

        # If user clicks 'Yes'
        if confirm_msg.clickedButton() == yes_button:
            # Create the information QMessageBox manually
            finish_msg = QtWidgets.QMessageBox(self)
            finish_msg.setWindowTitle("Finished")
            finish_msg.setText("Finished Generating Synthetic Data")
            finish_msg.setIcon(QtWidgets.QMessageBox.Information)

            # Add the 'Ok' button
            ok_button = finish_msg.addButton(QtWidgets.QMessageBox.Ok)
            ok_button.setStyleSheet("""
                background-color: #003333;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 5px;
                font-size: {font_sizes['button']}px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
            """)
            
            # Apply the same stylesheet
            finish_msg.setStyleSheet("""
                QMessageBox {
                    font-size: {font_sizes['content']}px;
                    font-weight: bold;
                    margin: 32px 32px;
                    font-family: 'Montserrat', sans-serif;
                }
                QPushButton:hover {
                    background-color: #005555;
                }
                QPushButton:pressed {
                    background-color: #002222;
                }
            """)

            # Show the message box
            finish_msg.exec_()

            # Emit the signal to clear the UI
            self.clearUI.emit()

    def handle_download_click(self):
        if self.output_zip_path and os.path.exists(self.output_zip_path):
            # Create the confirmation QMessageBox manually
            confirm_msg = QtWidgets.QMessageBox(self)
            confirm_msg.setWindowTitle("Download Confirmation")
            confirm_msg.setText("Do you want to download the generated zip file?")
            confirm_msg.setIcon(QtWidgets.QMessageBox.Question)

            # Add 'Yes' and 'No' buttons
            yes_button = confirm_msg.addButton(QtWidgets.QMessageBox.Yes)
            no_button = confirm_msg.addButton(QtWidgets.QMessageBox.No)
            yes_button.setStyleSheet("""
                background-color: #003333;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 5px;
                font-size: {font_sizes['button']}px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
            """)

            no_button.setStyleSheet("""
                background-color: #003333;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 5px;
                font-size: {font_sizes['button']}px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
            """)
            confirm_msg.setDefaultButton(no_button)

            # Apply the stylesheet for the confirmation box
            confirm_msg.setStyleSheet("""
                QMessageBox {
                    font-size: {font_sizes['content']}px;
                    font-weight: bold;
                    margin: 32px 32px;
                    font-family: 'Montserrat', sans-serif;
                }
                QPushButton:hover {
                    background-color: #005555;
                }
                QPushButton:pressed {
                    background-color: #002222;
                }
            """)

            # Show the confirmation message box
            confirm_msg.exec_()

            # If user clicks 'Yes', proceed with file saving
            if confirm_msg.clickedButton() == yes_button:
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

                        # Add custom Ok button and apply styles
                        ok_button = success_box.addButton(QtWidgets.QMessageBox.Ok)
                        success_box.setDefaultButton(ok_button)

                        ok_button.setStyleSheet("""
                            background-color: #003333;
                            color: white;
                            border: none;
                            padding: 5px 15px;
                            border-radius: 5px;
                            font-size: {font_sizes['button']}px;
                            font-weight: bold;
                            font-family: 'Montserrat', sans-serif;
                        """)

                        # Show success message box
                        success_box.exec_()

                    except Exception as e:
                        # Create and style the error message box
                        error_box = QtWidgets.QMessageBox(self)
                        error_box.setIcon(QtWidgets.QMessageBox.Warning)
                        error_box.setWindowTitle('Error')
                        error_box.setText(f"Error saving file: {str(e)}")

                        # Add custom Ok button and apply styles
                        ok_button = error_box.addButton(QtWidgets.QMessageBox.Ok)
                        error_box.setDefaultButton(ok_button)

                        ok_button.setStyleSheet("""
                            background-color: #003333;
                            color: white;
                            border: none;
                            padding: 5px 15px;
                            border-radius: 5px;
                            font-size: {font_sizes['button']}px;
                            font-weight: bold;
                            font-family: 'Montserrat', sans-serif;
                        """)


                        # Show error message box
                        error_box.exec_()

    def handle_remove_click(self):
        # Create the confirmation QMessageBox manually
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle('Confirmation')
        message_box.setText("Are you sure you want to remove the output file?")
        
        # Add custom Yes and No buttons
        yes_button = message_box.addButton(QtWidgets.QMessageBox.Yes)
        no_button = message_box.addButton(QtWidgets.QMessageBox.No)
        
        # Set the default button (No button)
        message_box.setDefaultButton(no_button)

        # Apply custom styles directly to the buttons
        yes_button.setStyleSheet("""
            background-color: #003333;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 5px;
            font-size: {font_sizes['button']}px;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
        """)

        no_button.setStyleSheet("""
            background-color: #003333;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 5px;
            font-size: {font_sizes['button']}px;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
        """)


        # Execute the message box and capture the user's response
        reply = message_box.exec_()

        # Handle the response
        if reply == QtWidgets.QMessageBox.Yes:
            # Proceed with removing the output file
            print("Removing the output file...")
            if self.output_zip_path and os.path.exists(self.output_zip_path):
                try:
                    os.remove(self.output_zip_path)
                    self.output_zip_path = None
                    self.setVisible(False)
                    
                    # Show success message after removal
                    success_msg = QtWidgets.QMessageBox(self)
                    success_msg.setWindowTitle("Success")
                    success_msg.setText("Output file has been removed")
                    success_msg.setIcon(QtWidgets.QMessageBox.Information)
                    
                    # Add Ok button and apply custom styles
                    ok_button = success_msg.addButton(QtWidgets.QMessageBox.Ok)
                    success_msg.setDefaultButton(ok_button)

                    ok_button.setStyleSheet("""
                        background-color: #003333;
                        color: white;
                        border: none;
                        padding: 5px 15px;
                        border-radius: 5px;
                        font-size: {font_sizes['button']}px;
                        font-weight: bold;
                        font-family: 'Montserrat', sans-serif;
                    """)


                    success_msg.exec_()

                except Exception as e:
                    # Show error message if removal fails
                    error_msg = QtWidgets.QMessageBox(self)
                    error_msg.setWindowTitle("Error")
                    error_msg.setText(f"Error removing file: {str(e)}")
                    error_msg.setIcon(QtWidgets.QMessageBox.Warning)

                    # Add Ok button and apply custom styles
                    ok_button = error_msg.addButton(QtWidgets.QMessageBox.Ok)
                    error_msg.setDefaultButton(ok_button)

                    ok_button.setStyleSheet("""
                        background-color: #003333;
                        color: white;
                        border: none;
                        padding: 5px 15px;
                        border-radius: 5px;
                        font-size: {font_sizes['button']}px;
                        font-weight: bold;
                        font-family: 'Montserrat', sans-serif;
                    """)

                    error_msg.exec_()

        else:
            # Do nothing if 'No' is selected
            print("Canceling the removal.")


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
                font-size: {font_sizes['content']}px;
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
                font-size: {font_sizes['button']}px;
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
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = OutputWidget()
    window.show()
    sys.exit(app.exec_())
