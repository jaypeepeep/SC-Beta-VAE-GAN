"""
Program Title: Workplace Page
Programmer/s:
- Alpapara, Nichole N.
- Lagatuz, John Patrick D.
- Peroche, John Mark P.
- Torreda, Kurt Denver P.
Description: The Workplace page servers as the main page when the user opens the system. It provides a user-friendly
interface for uploading files and generating synthetic data for the user. It composes of several widgets including
Input, File Preview, Process Log, Output, and Results. This page runs through the GenerateDataWorker module and is
connected using a thread so that the model will run smoothly to the user interface. The latter was setup using the
setupUi function. Moreover, several libraries were utilized such as numpy for numerical operations, pandas for data
manipulation, tensorflow, keras, and sklearn for model training.
Date Added: June 20, 2024
Last Date Modified: December 11, 2024

"""

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QColor
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_container_widget import FileContainerWidget
from components.widget.file_preview_widget import FilePreviewWidget
from components.widget.model_widget import ModelWidget
from components.widget.process_log_widget import ProcessLogWidget
from components.widget.output_widget import OutputWidget
from components.widget.spin_box_widget import SpinBoxWidget
from components.button.DragDrop_Button import DragDrop_Button
from components.widget.result_preview_widget import SVCpreview
from model import scbetavaegan
import os
import time
import shutil
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
from PyQt5.QtCore import QThread, pyqtSignal
import traceback
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from glob import glob
import re
from PyQt5.QtWidgets import QApplication
from pages.worker.generator import GenerateDataWorker
class Workplace(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Workplace, self).__init__(parent)
        self.uploaded_files = []
        self.setupUi()
        self.worker = None
        self.has_files = False

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setAlignment(QtCore.Qt.AlignTop)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.setFont(font)

        # Create a scroll area
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        # Create a container widget for the scroll area content
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)

        # Set a size policy for the scroll widget that allows it to shrink
        self.scroll_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )

        # Add the scroll area to the main layout
        self.scroll_area.setWidget(self.scroll_widget)
        self.gridLayout.addWidget(self.scroll_area)

        # Call functions to set up collapsible components
        self.setup_input_collapsible()
        self.setup_preview_collapsible()
        self.setup_model_collapsible()
        self.setup_process_log_collapsible()
        self.setup_output_collapsible()
        self.setup_result_collapsible()

        # Generate Synthetic Data button
        button_layout = QtWidgets.QVBoxLayout()
        self.generate_data_button = QtWidgets.QPushButton(
            "Generate Synthetic Data", self
        )
        self.generate_data_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 10px; 
                font-weight: 600; 
                padding: 10px 20px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
            """
        )
        self.generate_data_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )  # put the button at the bottom
        self.generate_data_button.clicked.connect(self.on_generate_data)

        button_layout.addWidget(
            self.generate_data_button, alignment=QtCore.Qt.AlignCenter
        )

        # spacer = QtWidgets.QSpacerItem(
        #     8, 8, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        # )
        # button_layout.addItem(spacer)

        # Adding the button to the main layout
        self.gridLayout.addLayout(button_layout, 1, 0)

    def train_vae(self):
        confirmation = self.model_widget.create_custom_message_box(
            title="Train SC-β-VAE-GAN",
            message=f"Are you sure you want to train a new model?",
        )

        # Proceed only if the user confirms with 'Yes'
        if confirmation:
            self.process_log_widget.clear()
            self.model_widget.uncheck_checkbox()
            self.svc_preview.clear()
            self.collapsible_widget_output.toggle_container(False)
            self.collapsible_widget_result.toggle_container(False)
            self.collapsible_widget_process_log.toggle_container(True)

            # Disable the generate button and change text
            self.generate_data_button.setEnabled(False)
            self.generate_data_button.setText("Generating...")

            # Create and start the worker thread
            self.worker = GenerateDataWorker(self)
            self.worker.set_model(None)
            self.num_augmentations = self.model_widget.slider_widget.getValue()
            self.worker.set_num_augmentations(self.num_augmentations)

            # Connect signals
            self.worker.generation_complete.connect(self.on_generation_finished)
            self.worker.finished.connect(self.on_generation_complete)
            self.worker.error.connect(self.on_generation_error)
            self.worker.progress.connect(
                self.logger.info
            )  # Connect directly to logger.info
            self.worker.metrics.connect(self.on_generation_results)

            # Start the thread
            self.worker.start()

    def on_generate_data(self):
        self.selected_model = self.model_widget.current_checked_file

        if self.has_files is False:
            self.show_error("Please upload a file first")
        elif self.selected_model == None:
            self.show_error(
                "Please select a pre-trained model first or train your own model"
            )
        elif self.has_files is True and self.selected_model != None:
            self.process_log_widget.clear()
            self.svc_preview.clear()
            self.collapsible_widget_output.toggle_container(False)
            self.collapsible_widget_result.toggle_container(False)
            self.collapsible_widget_process_log.toggle_container(True)
            if self.selected_model == "EMOTHAW.h5":
                self.svc_preview.add_graph_containers()
            # Disable the generate button and change text
            self.generate_data_button.setEnabled(False)
            self.generate_data_button.setText("Generating...")

            # Create and start the worker thread
            self.worker = GenerateDataWorker(self)
            self.worker.set_model(self.selected_model)
            self.num_augmentations = self.model_widget.slider_widget.getValue()
            self.worker.set_num_augmentations(self.num_augmentations)

            # Connect signals
            self.worker.error.connect(self.on_generation_error)
            self.worker.progress.connect(
                self.logger.info
            )  # Connect directly to logger.info
            self.worker.generation_complete.connect(self.on_generation_finished)
            self.worker.finished.connect(self.on_generation_complete)
            self.worker.metrics.connect(self.on_generation_results)

            # Start the worker
            self.worker.start()

    def on_generation_complete(self):
        # Re-enable the generate button
        self.generate_data_button.setEnabled(True)
        self.generate_data_button.setText("Generate Synthetic Data")

        self.model_widget.refresh_file_list()

        # Clean up
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def on_generation_finished(self):
        # Disable the generate button and change text
        self.generate_data_button.setEnabled(False)
        self.generate_data_button.setText("Calculating Results...")

        self.update_output_file_display(self.worker.all_augmented_filepaths)
        self.update_original_absolute_file_display(self.worker.original_absolute_files)
        self.output_widget.set_zip_path(self.worker.augmented_zip_filepath)

        # Expand relevant sections
        self.collapsible_widget_output.toggle_container(True)
        self.collapsible_widget_result.toggle_container(True)

    def on_generation_results(self, results):
        if results == "NRMSE":
            self.svc_preview.add_result_text(
                "Normalized Root Mean Square Error (NRMSE)"
            )
            self.svc_preview.add_result_text(
                f"\tOverall Average NRMSE: {self.worker.overall_avg_nrmse:.4f}"
            )
        elif results == "PHDS":
            self.svc_preview.add_result_text("\nPost-Hoc Discriminative Score (PHDS)")
            self.svc_preview.add_result_text(
                f"\tMean accuracy: {self.worker.mean_accuracy:.4f} (±{self.worker.std_accuracy:.4f})"
            )
        elif results == "PHPS":
            self.svc_preview.add_result_text("\nPost-Hoc Predictive Score (PHPS)")
            self.svc_preview.add_result_text(
                f"\tMean MAPE: {self.worker.mean_mape * 100:.2f}%"
            )
            self.svc_preview.add_result_text(
                f"\tStandard Deviation of MAPE: {self.worker.std_mape * 100:.2f}%"
            )

    def on_generation_error(self, error_message):
        # Re-enable the generate button
        self.generate_data_button.setEnabled(True)
        self.generate_data_button.setText("Generate Synthetic Data")

        # Show error message
        self.logger.error(f"Error during generation: {error_message}")

        # Clean up
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

        QMessageBox.critical(
            self,
            "Generation Error",
            f"An error occurred during data generation:\n{error_message}",
            QMessageBox.Ok,
        )

    def show_error(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(message)
        msg.setStyleSheet("""
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
        if message == "Please upload a file first":
            msg.setWindowTitle("File Upload Error")
        else:
            msg.setWindowTitle("Model Selection Error")

        # Set custom icon
        icon = QIcon("icon/icon.ico")
        msg.setWindowIcon(icon)

        msg.exec_()

    def update_file_scroll_area(self):
        """Update the scroll area's height based on the visibility of the DragDrop_Button."""
        if self.file_upload_widget.isVisible():
            self.file_scroll_area.setMinimumHeight(140)  # Reset to 0 if visible
        else:
            self.file_scroll_area.setMinimumHeight(300)  # Expand if not visible
        
    def setup_input_collapsible(self):
        """Set up the 'Input' collapsible widget and its contents."""
        font = QtGui.QFont()
        font.setPointSize(20)

        # Call the collapsible widget component for Input
        self.collapsible_widget_input = CollapsibleWidget("Input", self)
        self.scroll_layout.addWidget(self.collapsible_widget_input)

        # Add the FileUploadWidget
        self.file_upload_widget = DragDrop_Button(self)
        self.file_upload_widget.file_uploaded.connect(
            self.update_file_display
        )  # Connect the signal
        self.collapsible_widget_input.add_widget(self.file_upload_widget)

        # Add "Add More Files" button to Input collapsible widget
        self.add_file_button = QtWidgets.QPushButton("Add More Files", self)
        self.add_file_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 10px; 
                font-weight: 600; 
                padding: 10px 20px;
                margin-left: 15px; 
                margin-right: 15px; 
                border-radius: 5px; 
                border: none;
            }
            QPushButton:hover {
                background-color: #005555;  /* Change this to your desired hover color */
            }
            """
        )
        self.add_file_button.setFont(font)
        self.add_file_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.add_file_button.clicked.connect(self.add_more_files)
        self.scroll_layout.addWidget(self.add_file_button)

        # Create a scrollable area to hold the file widgets
        self.file_scroll_area = QtWidgets.QScrollArea(self)
        self.file_scroll_area.setWidgetResizable(True)
        self.file_scroll_area.setMinimumHeight(70)
        self.scroll_layout.addWidget(self.file_scroll_area)
        

        # Connect collapsible state or visibility changes
        self.file_upload_widget.file_uploaded.connect(self.update_file_scroll_area)

        # Create a container to hold the file widgets and its layout
        self.file_container_widget = QtWidgets.QWidget(self)
        self.file_container_layout = QtWidgets.QVBoxLayout(self.file_container_widget)
        self.file_container_layout.setSpacing(0)
        self.file_container_layout.setContentsMargins(0, 0, 0, 0)

        # Ensure the layout is aligned to the top
        self.file_container_layout.setAlignment(QtCore.Qt.AlignTop)

        # Add the file container widget to the scroll area
        self.file_scroll_area.setWidget(self.file_container_widget)
        self.collapsible_widget_input.add_widget(self.file_scroll_area)
                
        # Initially hide other components
        self.file_upload_widget.setVisible(True)
        self.show_other_components(False)

        # Open the collapsible widget by default
        self.collapsible_widget_input.toggle_container(True)

    def show_other_components(self, show=True):
        """Show or hide other components based on file upload."""
        self.add_file_button.setVisible(show)
        self.file_container_widget.setVisible(show)

    def setup_model_collapsible(self):
        self.collapsible_model_container = CollapsibleWidget("Models", self)
        self.scroll_layout.addWidget(self.collapsible_model_container)
        self.model_widget = ModelWidget(self)
        self.model_widget.train_button.clicked.connect(self.train_vae)
        self.selected_model = self.model_widget.current_checked_file
        self.num_augmentations = self.model_widget.slider_widget.getValue()
        self.collapsible_model_container.add_widget(self.model_widget)

    def setup_preview_collapsible(self):
        self.collapsible_widget_preview = CollapsibleWidget("File Preview", self)
        self.scroll_layout.addWidget(self.collapsible_widget_preview)

        self.file_preview_widget = FilePreviewWidget(self)
        self.collapsible_widget_preview.add_widget(self.file_preview_widget)

    def setup_process_log_collapsible(self):
        self.collapsible_widget_process_log = CollapsibleWidget("Process Log", self)
        self.scroll_layout.addWidget(self.collapsible_widget_process_log)

        self.process_log_widget = ProcessLogWidget(self)
        self.logger = self.process_log_widget.get_logger()
        self.collapsible_widget_process_log.add_widget(self.process_log_widget)

    def setup_output_collapsible(self):
        # Add the Output Widget
        self.collapsible_widget_output = CollapsibleWidget("Output", self)
        self.scroll_layout.addWidget(self.collapsible_widget_output)
        self.output_widget = OutputWidget(self)
        self.collapsible_widget_output.add_widget(self.output_widget)

    def setup_result_collapsible(self):
        """Set up the 'Result' collapsible widget and its contents."""

        # Call collapsible widget for Result
        self.collapsible_widget_result = CollapsibleWidget("Result", self)
        self.scroll_layout.addWidget(self.collapsible_widget_result)

        self.svc_preview = SVCpreview(self, mode="workplace")
        self.collapsible_widget_result.add_widget(self.svc_preview)

    def handle_checkbox_click(self, filename, state):
        if state == QtCore.Qt.Checked:
            self.selected_filename = filename
            print(f"Selected file: {self.selected_filename}")
        else:
            if self.selected_filename == filename:
                self.selected_filename = None
            print(f"Deselected file: {filename}")

    def handle_file_removal(self, file_path, file_name):
        """Handle the file removal logic when a file is removed."""
        if file_path in self.uploaded_files:
            # Remove the file from the uploaded_files list
            self.uploaded_files.remove(file_path)
            print(
                f"Removed file: {file_name}, remaining files: {self.uploaded_files}"
            )  # Debug statement

            # Update the UI to reflect the removal
            for i in reversed(range(self.file_container_layout.count())):
                widget = self.file_container_layout.itemAt(i).widget()
                if (
                    isinstance(widget, FileContainerWidget)
                    and widget.file_name == file_name
                ):
                    widget.remove_file_signal.disconnect()  # Disconnect signal to avoid errors
                    self.file_container_layout.removeWidget(
                        widget
                    )  # Remove the widget from layout
                    widget.deleteLater()  # Schedule the widget for deletion
                    widget.setParent(None)  # Detach widget from its parent
                    break  # Exit after removing the specific file container

            # If no more files, show the file upload widget again
            if not self.uploaded_files:
                self.show_other_components(False)
                self.file_upload_widget.setVisible(True)
                self.file_preview_widget.clear()

            # Update the file container layout to reflect the changes
            self.file_container_layout.update()
            self.has_files = bool(self.uploaded_files)
            if self.has_files == False:
                self.clear_all_ui()

    def update_file_display(self, new_uploaded_files):
        """Update the display of files based on newly uploaded files."""
        # Append new files to the existing list, avoiding duplicates
        for file_path in new_uploaded_files:
            if file_path not in self.uploaded_files:
                self.uploaded_files.append(file_path)

        print("Uploaded files:", self.uploaded_files)  # Debugging output

        self.has_files = bool(self.uploaded_files)
        self.show_other_components(self.has_files)

        # Hide the file upload widget if files are uploaded
        self.file_upload_widget.setVisible(not self.has_files)

        # Clear existing widgets in the file container layout
        for i in reversed(range(self.file_container_layout.count())):
            widget = self.file_container_layout.itemAt(i).widget()
            if widget is not None:
                widget.remove_file_signal.disconnect()  # Disconnect signal to avoid errors
                widget.deleteLater()  # Schedule widget deletion
                self.file_container_layout.removeWidget(widget)

        # Re-add file containers for each uploaded file and update preview
        for index, file_path in enumerate(self.uploaded_files):
            file_name = os.path.basename(file_path)

            # Verify the file still exists before displaying it
            if os.path.exists(file_path):
                new_file_container = FileContainerWidget(file_path, self)
                new_file_container.hide_download_button()
                new_file_container.remove_file_signal.connect(
                    self.handle_file_removal
                )  # Connect remove signal
                self.file_container_layout.addWidget(new_file_container)

                # Check if this is the first file
                if index == 0:  # This means it's the first file
                    # Display the file content in the file preview widget
                    self.file_preview_widget.display_file_contents(file_path)

        self.file_preview_widget.set_uploaded_files(self.uploaded_files)
        self.svc_preview.set_uploaded_files(self.uploaded_files)

        # Automatically expand the preview collapsible widget if there are files
        if self.has_files:
            self.collapsible_model_container.toggle_container(True)
            self.collapsible_widget_preview.toggle_container(True)

    def update_output_file_display(self, all_augmented_filepaths):
        """Update the display of files based on newly generated augmented files."""
        for index, file_path in enumerate(all_augmented_filepaths):
            # Verify the file still exists before displaying it
            if os.path.exists(file_path):
                new_output_file_container = FileContainerWidget(file_path, self)
                new_output_file_container.hide_remove_button()

                # Check if this is the first file
                if index == 0:  # This means it's the first file
                    self.svc_preview.display_file_contents(file_path, 1)
                    self.svc_preview.display_graph_contents(file_path, 1)
                    self.svc_preview.display_handwriting_contents(file_path, 1, mode="workplace")
                    self.svc_preview.display_table_contents(file_path, 1)

        self.svc_preview.set_augmented_files(all_augmented_filepaths)

        # Automatically expand the output collapsible widget
        self.collapsible_widget_output.toggle_container(True)

    def update_original_absolute_file_display(self, original_absolute_files):
        """Update the display of original absolute files based on newly generated augmented files."""
        for index, file_path in enumerate(original_absolute_files):
            # Verify the file still exists before displaying it
            if os.path.exists(file_path):
                if index == 0:  # This means it's the first file
                    self.svc_preview.display_file_contents(file_path, 0)
                    self.svc_preview.display_graph_contents(file_path, 0)
                    self.svc_preview.display_handwriting_contents(file_path, 0, mode="workplace")
                    self.svc_preview.display_table_contents(file_path, 0)

        self.svc_preview.set_original_absolute_files(original_absolute_files)

    def add_more_files(self):
        self.file_upload_widget.open_file_dialog()

    def get_image_path(self, image_name):
        path = os.path.join(os.path.dirname(__file__), "..", "icon", image_name)
        return path

    def clear_all_ui(self):
        # Clear uploaded files
        self.uploaded_files = []

        # Reset file upload widget
        self.file_upload_widget.setVisible(True)
        self.show_other_components(False)

        # Clear file containers
        for i in reversed(range(self.file_container_layout.count())):
            widget = self.file_container_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Clear previews and logs
        self.file_preview_widget.clear()
        self.process_log_widget.clear()
        self.svc_preview.clear()
        self.model_widget.uncheck_checkbox()
        self.model_widget.slider_widget.resetValue()

        # Collapse all widgets except Input
        self.collapsible_widget_preview.toggle_container(False)
        self.collapsible_model_container.toggle_container(False)
        self.collapsible_widget_process_log.toggle_container(False)
        self.collapsible_widget_output.toggle_container(False)
        self.collapsible_widget_result.toggle_container(False)


if __name__ == "__main__":

    app = QtWidgets.QApplication([])
    window = Workplace()
    window.show()
    app.exec_()
