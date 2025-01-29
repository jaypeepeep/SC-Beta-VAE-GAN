import subprocess
import requests
import os
import sys
import time
import shutil
import zipfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import  QTimer
from PyQt5.QtWidgets import QVBoxLayout, QScrollArea, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from components.button.handwriting_button import handwritingButton
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_preview_widget import FilePreviewWidget
from components.widget.process_log_widget import ProcessLogWidget
from components.widget.output_widget import OutputWidget
from components.widget.file_container_widget import FileContainerWidget
from components.widget.plot_container_widget import PlotContainerWidget
from components.widget.spin_box_widget import SpinBoxWidget
from components.widget.result_preview_widget import SVCpreview
from pages.writer.trainer import ModelTrainingThread
from model.scbetavaegan_pentab import (
    calculate_nrmse,
    post_hoc_discriminative_score,
)

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.drawing_done = False
        self.flask_process = None
        self.file_list = []
        self.uploads_dir = os.path.abspath("uploads")
        self.threads = []
        self.setupUi()

        if not os.path.exists(self.uploads_dir):
            os.makedirs(self.uploads_dir)

    def setupUi(self):
        """Initial setup for the drawing page or Flask app depending on the file_list state."""
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setContentsMargins(50, 0, 50, 50)

        # Initialize Process Log Widget
        self.process_log_widget = ProcessLogWidget(self)
        self.logger = self.process_log_widget.get_logger()
        self.layout.addWidget(self.process_log_widget)

        # Initialize Output Widget
        self.output_widget = OutputWidget(self)
        self.layout.addWidget(self.output_widget)

        # Initialize Result Preview Widget
        self.result_preview_widget = SVCpreview(self)
        self.layout.addWidget(self.result_preview_widget)

        # Set widgets initially collapsed
        self.process_log_widget.setVisible(False)
        self.output_widget.setVisible(False)
        self.result_preview_widget.setVisible(False)

        # Check if there is existing handwriting data (i.e., file_list is not empty)
        if self.file_list:
            self.show_embedded_browser()
        else:
            self.show_drawing_page()

    def clear_layout(self):
        """Clear the current layout and any child layouts."""
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset references to deleted widgets
        self.process_log_widget = None
        self.result_preview_widget = None
        self.output_widget = None

    def clear_layout_recursively(self, layout):
        """Recursively clear all widgets and child layouts in the given layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            if item.layout():
                self.clear_layout_recursively(item.layout())
            del item

    def show_drawing_page(self):
        """Show the drawing page layout with the Draw and Handwrite button."""
        self.clear_layout()

        # Create a layout for the text
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.setAlignment(QtCore.Qt.AlignCenter)
        top_layout.setContentsMargins(0, 20, 0, 20)

        top_text = QtWidgets.QLabel("Draw and Handwrite", self)
        top_text.setAlignment(QtCore.Qt.AlignCenter)
        top_text.setStyleSheet("font-size: 30px; font-weight: bold; color: #033; ")
        top_layout.addWidget(top_text)
        self.layout.addLayout(top_layout)

        drawButton = handwritingButton(self)
        drawButton.setContentsMargins(50, 20, 50, 50)
        self.layout.addWidget(drawButton)

        drawButton.clicked.connect(self.show_confirmation_dialog)

    def show_confirmation_dialog(self):
        """Show a confirmation dialog before proceeding to the drawing page."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Proceed to Handwriting & Drawing")
        message_box.setText("Do you want to start drawing and handwriting?")
        message_box.setStandardButtons(
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        message_box.setDefaultButton(QtWidgets.QMessageBox.Ok)
        layout = message_box.layout()
        layout.setContentsMargins(20, 20, 20, 20)  
        layout.setSpacing(10)  
        message_box.setStyleSheet("""
            QMessageBox {
                font-size: 12px;
                font-weight: bold;
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
        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Ok:
            self.run_flask_app()

    def run_flask_app(self):
        """Run the Flask app located in components/canvas/app.py and open it in the embedded browser."""
        flask_app_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../components/canvas/app.py")
        )

        # Run the Flask app as a subprocess
        self.flask_process = subprocess.Popen(["python", flask_app_path])
        QtCore.QTimer.singleShot(3000, self.show_embedded_browser)

    def show_embedded_browser(self):
        """Show the Flask app inside the Handwriting page using QWebEngineView."""
        self.clear_layout()

        # Create a QWebEngineView and load the Flask app's URL
        self.webview = QWebEngineView(self)
        self.webview.setUrl(QtCore.QUrl("http://127.0.0.1:5000"))
        self.layout.addWidget(self.webview)

        self.webview.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        QtCore.QTimer.singleShot(
            5000, self.check_drawing_done
        ) 

    def check_drawing_done(self):
        """Periodically check if the drawing is done by querying Flask."""
        try:
            response = requests.get("http://127.0.0.1:5000/check_upload")
            if response.status_code == 200:
                data = response.json()
                filename = data.get("filename")
                if filename.endswith(".svc"): 
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self.folder_name = f"HandwritingData_{timestamp}"
                    self.handwriting_data_dir = os.path.join(
                        self.uploads_dir, self.folder_name
                    )
                    os.makedirs(self.handwriting_data_dir, exist_ok=True)
                    source_file = os.path.join(self.uploads_dir, filename)
                    dest_file = os.path.join(self.handwriting_data_dir, filename)
                    shutil.copy(source_file, dest_file)
                    self.show_done_page(filename)
                    self.svc_preview.set_uploaded_files(self.file_list)
                    if filename not in self.file_list:
                        self.file_list.append(filename)
                        print("File list:", self.file_list)
                        if hasattr(self, "file_preview_widget"):
                            self.file_preview_widget.set_uploaded_files(self.file_list)
                else:
                    self.process_log_widget.append_log(f"Invalid file type: {filename}")
            else:
                QtCore.QTimer.singleShot(5000, self.check_drawing_done)
        except requests.ConnectionError:
            QtCore.QTimer.singleShot(5000, self.check_drawing_done)

    def show_done_page(self, filename, is_file_removal=False):
        """Show the page after the drawing is completed."""
        if not is_file_removal and filename not in self.file_list:
            self.file_list.append(filename)
        self.clear_layout()

        # Create a scroll area to wrap the collapsible content
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        # Create a widget that will be placed inside the scroll area
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setAlignment(QtCore.Qt.AlignTop)

        # Create a scrollable widget
        sub_area = QScrollArea()
        sub_area.setWidgetResizable(True)

        # Create a container for the scroll area
        sub_container = QWidget()
        sub_container.setMaximumHeight(300)
        sub_layout = QVBoxLayout(sub_container)

        # Add file containers to the scrollable layout
        for file in self.file_list:
            file_container = FileContainerWidget(file, self)
            file_container.remove_file_signal.connect(self.remove_file)
            sub_layout.addWidget(file_container)

        sub_area.setWidget(sub_container)

        # Add the scroll area to the collapsible widget
        scroll_area.setWidget(scroll_widget)
        self.layout.addWidget(scroll_area)

        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        scroll_layout.addWidget(self.collapsible_widget)
        self.collapsible_widget.toggle_container(True)

        added_files = set()
        for file in self.file_list:
            if file not in added_files:
                file_container = FileContainerWidget(file, self)
                file_container.remove_file_signal.connect(self.remove_file)
                self.collapsible_widget.add_widget(file_container)
                added_files.add(file)

        # Add the dropdown (QComboBox) for selecting a file to plot
        self.file_dropdown = QtWidgets.QComboBox(self)
        self.file_dropdown.setStyleSheet(
            """
            QComboBox {
                background-color: #033;  
                color: white; 
                font-weight: bold;           
                font-family: Montserrat; 
                font-size: 14px;        
                padding: 10px;            
                border: 2px solid #033;  
                border-radius: 5px;      
            }

            /* Dropdown arrow styling */
            QComboBox::drop-down {
                border: none;
            }

            /* Dropdown arrow icon */
            QComboBox::down-arrow {
                image: url(arrow_down_icon.png); 
                width: 14px;
                height: 14px;
            }

            /* Styling for the dropdown items */
            QComboBox QAbstractItemView {
                background-color: white;   
                color: #033;                 
                border: 1px solid #033;    
                font-family: Montserrat;
                font-size: 14px;
        }"""
        )
        self.file_dropdown.addItems(self.file_list)
        self.file_dropdown.currentIndexChanged.connect(self.on_file_selected)

        # Add the dropdown to the collapsible widget
        self.collapsible_widget.add_widget(self.file_dropdown)

        # Add the plot container widget
        self.plot_container = PlotContainerWidget(self)
        self.collapsible_widget.add_widget(self.plot_container)

        # Initially load the plot for the first file in the list
        if self.file_list:
            self.plot_container.loadPlot(self.file_list[0])

        # Add the slider widget directly to the collapsible widget
        self.spin_box_widget = SpinBoxWidget(1)
        self.collapsible_widget.add_widget(self.spin_box_widget)

        # Add "Draw More" and "Clear All" buttons inside the collapsible widget
        button_layout = QtWidgets.QHBoxLayout()

        self.draw_more_button = QtWidgets.QPushButton("Draw More", self)
        self.draw_more_button.setStyleSheet(
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
        self.draw_more_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.draw_more_button.clicked.connect(self.run_flask_app)

        self.clear_all_button = QtWidgets.QPushButton("Clear All", self)
        self.clear_all_button.setStyleSheet(
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
        self.clear_all_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.clear_all_button.clicked.connect(self.clear_all_drawings)

        # Add the buttons to the button layout
        button_layout.addWidget(self.draw_more_button)
        button_layout.addWidget(self.clear_all_button)

        # Add the button layout to the collapsible widget
        button_widget = QtWidgets.QWidget()  # Wrap buttons in a QWidget
        button_widget.setLayout(button_layout)
        self.collapsible_widget.add_widget(button_widget)

        # Add the File Preview Widget
        self.collapsible_widget_file_preview = CollapsibleWidget("File Preview", self)
        scroll_layout.addWidget(self.collapsible_widget_file_preview)
        self.file_preview_widget = FilePreviewWidget(self)
        self.file_preview_widget.set_uploaded_files(self.file_list)
        self.collapsible_widget_file_preview.add_widget(self.file_preview_widget)

        # Add the Process Log Widget
        self.collapsible_widget_process_log = CollapsibleWidget("Process Log", self)
        scroll_layout.addWidget(self.collapsible_widget_process_log)
        self.process_log_widget = ProcessLogWidget(self)
        self.collapsible_widget_process_log.add_widget(self.process_log_widget)

        # Add the Output Widget
        self.collapsible_widget_output = CollapsibleWidget("Output", self)
        scroll_layout.addWidget(self.collapsible_widget_output)
        self.output_widget = OutputWidget(self)
        self.collapsible_widget_output.add_widget(self.output_widget)

        # Call the collapsible widget component for result
        self.collapsible_widget_result = CollapsibleWidget("Result", self)
        scroll_layout.addWidget(self.collapsible_widget_result)
        self.svc_preview = SVCpreview(input=filename)
        self.collapsible_widget_result.add_widget(self.svc_preview)

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

        spacer = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        button_layout.addItem(spacer)

        # Adding the button to the main layout
        scroll_layout.addLayout(button_layout)

        # Automatically open file preview widget after 2 secs
        QTimer.singleShot(2000, lambda: self.collapsible_widget_file_preview.toggle_container(True))
    

    def remove_file(self, file_path, file_name):
        """Handle the removal of a file from the file list."""
        if file_path in self.file_list:
            self.file_list.remove(file_path)
            self.process_log_widget.append_log(f"Removed file: {file_name}")

            # If no files left, reset to the initial drawing page
            if not self.file_list:
                self.reset_state()
                return
            
            # If there's only one file left, properly reset and show done page
            if len(self.file_list) == 1:
                self.clear_layout()
                self.show_done_page(self.file_list[0], is_file_removal=True)
                return

            # For two or more remaining files, just update the UI
            layout = self.collapsible_widget.collapsible_layout
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, FileContainerWidget) and widget.file_path == file_path:
                    layout.removeWidget(widget)
                    widget.setParent(None)
                    widget.deleteLater()
                    break

            # Update the file dropdown
            current_index = self.file_dropdown.currentIndex()
            self.file_dropdown.clear()
            self.file_dropdown.addItems(self.file_list)
            if current_index >= len(self.file_list):
                current_index = len(self.file_list) - 1
            self.file_dropdown.setCurrentIndex(current_index)

            # Update the file preview widget
            if hasattr(self, "file_preview_widget"):
                self.file_preview_widget.set_uploaded_files(self.file_list)

    def update_file_display(self):
        """Update the display of files in the UI after removal."""
        self.clear_layout()
        if self.file_list:
            self.show_done_page(self.file_list[-1])
        else:
            self.reset_state() 

    def on_generate_data(self):
        """Start processing the selected .svc files."""
        uploads_dir = "uploads"
        num_augmented_files = self.spin_box_widget.number_input.value()
        epochs = 350
        self.svc_preview.remove_graph_containers()

        if not self.file_list:
            self.process_log_widget.append_log("No files available for processing.")
            return

        self.process_log_widget.setVisible(True)
        self.collapsible_widget_process_log.toggle_container(True)
        self.generate_data_button.setEnabled(False)
        self.generate_data_button.setText("Generating...")
        self.svc_preview.add_graph_containers()

        file_count = len(self.file_list)
        self.process_log_widget.append_log(
            f"Starting data generation for {file_count} file(s)..."
        )

        for selected_file in self.file_list:
            if not selected_file.endswith(".svc"):
                self.process_log_widget.append_log(
                    f"Skipping non-.svc file: {selected_file}"
                )
                continue

            # Start a new thread for each file
            thread = ModelTrainingThread(
                self.handwriting_data_dir,
                self.file_list,
                uploads_dir,
                selected_file,
                num_augmented_files,
                epochs,
                logger=self.logger,
            )
            self.threads.append(thread)
            thread.log_signal.connect(self.process_log_widget.append_log)
            thread.zip_ready.connect(self.on_zip_ready)
            thread.metrics_ready.connect(self.on_metrics_ready)
            thread.finished.connect(self.on_thread_finished)
            thread.original_files_ready.connect(self.update_original_absolute_file_display)
            thread.augmented_files_ready.connect(self.update_output_file_display)
            thread.start()

        self.process_log_widget.append_log("All threads started, awaiting results...")

    def closeEvent(self, event):
        """Ensure the Flask app process and threads are killed when the main window is closed."""
        # Terminate the Flask process if running
        if self.flask_process:
            self.flask_process.terminate()

        # Stop all running threads
        for thread in self.threads:
            if thread.isRunning():
                thread.quit()
                thread.wait() 

        event.accept()

    def on_thread_finished(self):
        """Callback when a single file has finished processing."""
        self.process_log_widget.append_log("A file has finished processing.")

        # Check if all threads are done before re-enabling the button
        for thread in self.threads:
            if thread.isFinished():
                self.threads.remove(thread) 

        if not self.threads:
            self.process_log_widget.append_log("All files have finished processing.")
            self.generate_data_button.setEnabled(True)

    def on_zip_ready(self, zip_file_path):
        # Set the zip path for output widget
        if hasattr(self.output_widget, "set_zip_path"):
            QtCore.QMetaObject.invokeMethod(
                self.output_widget,
                "set_zip_path",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, zip_file_path),
            )
            self.output_widget.setVisible(True)
            self.collapsible_widget_output.toggle_container(True)

        self.collapsible_widget_result.toggle_container(True)

    def on_metrics_ready(self, metrics):
        """Update the results_text widget with the calculated metrics."""
        metrics_text = ""

        # Normalized Root Mean Square Error (NRMSE)
        if "Normalized Root Mean Square Error (NRMSE)" in metrics:
            overall_avg_nrmse = metrics["Normalized Root Mean Square Error (NRMSE)"]
            metrics_text += "Normalized Root Mean Square Error (NRMSE)\n"
            metrics_text += f"\tOverall Average NRMSE: {overall_avg_nrmse:.4f}\n\n"

        # Post-Hoc Discriminative Score (PHDS)
        if "Discriminative Mean Accuracy" in metrics and "Discriminative Accuracy Std" in metrics:
            mean_acc = metrics["Discriminative Mean Accuracy"]
            std_acc = metrics["Discriminative Accuracy Std"]
            metrics_text += "Post-Hoc Discriminative Score (PHDS)\n"
            metrics_text += f"\tMean accuracy: {mean_acc:.4f} (±{std_acc:.4f})\n\n"

        # Post-Hoc Predictive Score (PHPS)
        if "Mean MAPE" in metrics and "Standard Deviation of MAPE" in metrics:
            mean_mape = metrics["Mean MAPE"] * 100 
            std_mape = metrics["Standard Deviation of MAPE"] * 100
            metrics_text += "Post-Hoc Predictive Score (PHPS)\n"
            metrics_text += f"\tMean MAPE: {mean_mape:.2f}%\n"
            metrics_text += f"\tStandard Deviation of MAPE: {std_mape:.2f}%\n"

        self.svc_preview.results_text.setPlainText(metrics_text)


    def on_training_finished(self):
        """Callback when training and data generation is finished."""
        self.generate_data_button.setText("Generate Synthetic Data")
        self.generate_data_button.setEnabled(True)
        self.process_log_widget.append_log("Data generation finished.")

    def get_absolute_paths(self, directory, filenames):
        """
        Given a directory and a list of filenames, return a list of absolute paths.

        Args:
            directory (str): The base directory where the files are located.
            filenames (list): A list of filenames (relative paths).

        Returns:
            list: A list of absolute paths.
        """
        absolute_paths = []
        for filename in filenames:
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            absolute_paths.append(absolute_path)
        return absolute_paths

    def extract_paths_from_zip(self, zip_path, extract_to):
        """
        Extract the .svc files from a zip archive and return their absolute paths.

        Args:
            zip_path (str): Path to the zip file containing synthetic data.
            extract_to (str): Directory where the files will be extracted.

        Returns:
            list: A list of absolute paths to the extracted .svc files.
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract all .svc files to the specified directory
            zip_ref.extractall(extract_to)

        # Gather paths of all extracted .svc files
        svc_paths = [
            os.path.abspath(os.path.join(extract_to, file))
            for file in os.listdir(extract_to)
            if file.endswith(".svc")
        ]
        return svc_paths

    def update_output_file_display(self, augmented_files):
        """
        Update the display of files based on newly generated augmented files.
        """

        # Ensure paths are correctly set and the files exist
        for index, file_path in enumerate(augmented_files):
            if os.path.exists(file_path):
                if index == 0:
                    self.svc_preview.display_file_contents(file_path, 1)
                    self.svc_preview.display_graph_contents(file_path, 1)
                    self.svc_preview.display_handwriting_contents(file_path, 1)
                    self.svc_preview.display_table_contents(file_path, 1)

        self.svc_preview.set_augmented_files(augmented_files)

        # Automatically expand the output collapsible widget
        self.collapsible_widget_output.toggle_container(True)

    def update_original_absolute_file_display(self, original_absolute_files):
        """Update the display of original absolute files based on newly generated augmented files."""
        for index, file_path in enumerate(original_absolute_files):
            if os.path.exists(file_path):
                if index == 0:
                    self.svc_preview.display_file_contents(file_path, 0)
                    self.svc_preview.display_graph_contents(file_path, 0)
                    self.svc_preview.display_handwriting_contents(file_path, 0)
                    self.svc_preview.display_table_contents(file_path, 0)

        self.svc_preview.set_original_absolute_files(original_absolute_files)

    def calculate_metrics(self, original_file, synthetic_file):
        """Calculate and return the NRMSE, discriminative, and predictive scores."""
        original_data = pd.read_csv(
            original_file,
            sep=" ",
            names=[
                "x",
                "y",
                "timestamp",
                "pen_status",
                "pressure",
                "azimuth",
                "altitude",
            ],
        )
        synthetic_data = pd.read_csv(
            synthetic_file,
            sep=" ",
            names=[
                "x",
                "y",
                "timestamp",
                "pen_status",
                "pressure",
                "azimuth",
                "altitude",
            ],
        )

        # Compute NRMSE
        nrmse = calculate_nrmse(
            original_data[["x", "y"]].values, synthetic_data[["x", "y"]].values
        )

        # Compute Post-Hoc Discriminative Score (you can use the LSTM model for this)
        discriminative_score = post_hoc_discriminative_score(
            original_data, synthetic_data
        )

        # Compute Post-Hoc Predictive Score (LSTM-based predictive model)
        # predictive_score = post_hoc_predictive_score(original_data, synthetic_data)

        return {
            "nrmse": nrmse,
            "discriminative_score": discriminative_score,
            # "predictive_score": predictive_score
        }

    def show_reset_confirmation_dialog(self):
        """Show a confirmation dialog before resetting the state."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Discard and Retry")
        message_box.setText(
            "Are you sure you want to discard your current handwriting and start over?"
        )
        message_box.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        message_box.setDefaultButton(QtWidgets.QMessageBox.No)
        message_box.setStyleSheet("""
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
        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Yes:
            self.reset_state()

    def clear_all_drawings(self):
        """Clear all added files and reset the state."""
        self.file_list.clear()

        # Stop all running threads when clearing all drawings
        for thread in self.threads:
            if thread.isRunning():
                thread.quit()
                thread.wait()

        self.threads.clear()  # Clear the thread list after stopping them

        self.show_drawing_page()  # Go back to the initial drawing page

    def reset_state(self):
        """Reset the state and go back to the drawing page."""
        self.drawing_done = False
        self.file_list.clear()  # Clear file list when resetting
        self.show_drawing_page()

    def on_file_selected(self):
        """Update the plot when a different file is selected from the dropdown."""
        selected_file = self.file_dropdown.currentText()
        self.plot_container.loadPlot(selected_file)

    def closeEvent(self, event):
        """Ensure the Flask app process is killed when the main window is closed."""
        if self.flask_process:
            self.flask_process.terminate()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())