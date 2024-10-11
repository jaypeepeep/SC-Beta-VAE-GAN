import subprocess
import requests
import os
import sys
import time
import shutil
import zipfile
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
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
from pages.local import Local
from model.scbetavaegan_pentab import (
    upload_and_process_files,
    generate_augmented_datasets,
    load_pretrained_vae,
    download_augmented_data_as_integers
)

class ModelTrainingThread(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)  # Signal to send log updates

    def __init__(self, model, processed_data, data_frames, num_augmented_files, avg_data_points, scalers, original_data_frames, input_filenames, output_dir):
        super().__init__()
        self.model = model
        self.processed_data = processed_data
        self.data_frames = data_frames
        self.num_augmented_files = num_augmented_files
        self.avg_data_points = avg_data_points
        self.scalers = scalers
        self.original_data_frames = original_data_frames
        self.input_filenames = input_filenames
        self.output_dir = output_dir

    def run(self):
        self.log_signal.emit("Starting synthetic data generation...")
        
        # Generate augmented datasets
        augmented_datasets = generate_augmented_datasets(
            self.model, self.processed_data, self.data_frames, 
            self.num_augmented_files, self.avg_data_points
        )
        self.log_signal.emit("Augmented datasets generated successfully.")
        
        # Save the augmented datasets to the uploads folder
        download_augmented_data_as_integers(
            augmented_datasets, self.scalers, self.original_data_frames, self.input_filenames, directory=self.output_dir
        )
        
        # Zip the augmented data
        zip_path = os.path.join(self.output_dir, 'augmented_data.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        self.log_signal.emit(f"Augmented data saved and zipped at {zip_path}")
        self.finished.emit()

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.drawing_done = False  # State to check if done button was clicked
        self.flask_process = None  # To keep track of the Flask process
        self.current_filename = None
        self.file_list = []
        self.setupUi()

    def setupUi(self):
        """Initial setup for the drawing page or Flask app depending on the file_list state."""
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setContentsMargins(50, 0, 50, 50)

        # Check if there is existing handwriting data (i.e., file_list is not empty)
        if self.file_list:
            # If handwriting data exists, skip the drawing page and show the embedded browser
            self.show_embedded_browser()
        else:
            # If no handwriting data, show the drawing page with the button
            self.show_drawing_page()

    def clear_layout(self):
        """Clear the current layout and any child layouts."""
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            if item.layout():
                self.clear_layout_recursively(item.layout())
            del item

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

        # Add text
        top_text = QtWidgets.QLabel("Draw and Handwrite", self)
        top_text.setAlignment(QtCore.Qt.AlignCenter)
        top_text.setStyleSheet("font-size: 30px; font-weight: bold; color: #033; ")
        top_layout.addWidget(top_text)
        self.layout.addLayout(top_layout)

        # Create and add the handwriting button
        drawButton = handwritingButton(self)
        drawButton.setContentsMargins(50, 20, 50, 50)
        self.layout.addWidget(drawButton)

        # Connect the button's click events
        drawButton.clicked.connect(self.show_confirmation_dialog)
    def show_confirmation_dialog(self):
        """Show a confirmation dialog before proceeding to the drawing page."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Proceed to Handwriting & Drawing")
        message_box.setText(
            "Do you want to start drawing and handwriting?"
        )
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        message_box.setDefaultButton(QtWidgets.QMessageBox.Ok)

        # Apply stylesheet to customize button font size
        message_box.setStyleSheet("QPushButton { font-size: 14px; }")
    
        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Ok:
            self.run_flask_app()

    def run_flask_app(self):
        """Run the Flask app located in components/canvas/app.py and open it in the embedded browser."""
        flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../components/canvas/app.py'))
        
        # Run the Flask app as a subprocess
        self.flask_process = subprocess.Popen(['python', flask_app_path])

        # Display the embedded browser after a short delay to ensure Flask is running
        QtCore.QTimer.singleShot(5000, self.show_embedded_browser)

    def show_embedded_browser(self):
        """Show the Flask app inside the Handwriting page using QWebEngineView."""
        self.clear_layout()

        # Create a QWebEngineView and load the Flask app's URL
        self.webview = QWebEngineView(self)
        self.webview.setUrl(QtCore.QUrl("http://127.0.0.1:5000"))
        self.layout.addWidget(self.webview)

        # Ensure the webview resizes responsively
        self.webview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Poll Flask to check if drawing is done and file is uploaded
        QtCore.QTimer.singleShot(5000, self.check_drawing_done)  # Adjust the delay if necessary
            
    def check_drawing_done(self):
        """Periodically check if the drawing is done by querying Flask."""
        try:
            response = requests.get("http://127.0.0.1:5000/check_upload")
            if response.status_code == 200:
                data = response.json()
                filename = data.get('filename')
                self.show_done_page(filename)  # Pass the filename to the next page
                self.file_list.append(filename)
                if hasattr(self, 'file_preview_widget'):
                    self.file_preview_widget.set_uploaded_files(self.file_list)
            else:
                print("File not uploaded yet, retrying...")
                QtCore.QTimer.singleShot(5000, self.check_drawing_done)  # Retry after delay
        except requests.ConnectionError:
            print("Flask server not ready, retrying...")
            QtCore.QTimer.singleShot(5000, self.check_drawing_done)  # Retry after delay if connection failed

    def show_done_page(self, filename):
        """Show the page after the drawing is completed."""
        self.file_list.append(filename)  # Append the new filename to the list
        self.clear_layout()

        # Create a scroll area to wrap the collapsible content
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # Create a widget that will be placed inside the scroll area
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setAlignment(QtCore.Qt.AlignTop)
        ##
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
            sub_layout.addWidget(file_container)

        # Set the scrollable widget
        sub_area.setWidget(sub_container)

        # Add the scroll area to the collapsible widget
        ##
        # Add the scroll area to the main layout
        scroll_area.setWidget(scroll_widget)
        self.layout.addWidget(scroll_area)

        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        scroll_layout.addWidget(self.collapsible_widget)
        self.collapsible_widget.toggle_container(True)

        # Add a file container widget to the collapsible widget for each drawing added
        for file in self.file_list:
            file_container = FileContainerWidget(file, self)
            self.collapsible_widget.add_widget(file_container)

        # Add the dropdown (QComboBox) for selecting a file to plot
        self.file_dropdown = QtWidgets.QComboBox(self)
        self.file_dropdown.setStyleSheet("""
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
        }""")
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
        self.spin_box_widget = SpinBoxWidget(0)
        self.collapsible_widget.add_widget(self.spin_box_widget)

        # Add "Draw More" and "Clear All" buttons inside the collapsible widget
        button_layout = QtWidgets.QHBoxLayout()
        
        self.draw_more_button = QtWidgets.QPushButton("Draw More", self)
        self.draw_more_button.setStyleSheet("""
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 14px; 
                font-weight: 600; 
                padding: 10px 20px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
        """)
        self.draw_more_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.draw_more_button.clicked.connect(self.run_flask_app)
        
        self.clear_all_button = QtWidgets.QPushButton("Clear All", self)
        self.clear_all_button.setStyleSheet("""
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 14px; 
                font-weight: 600; 
                padding: 10px 20px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
        """)
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

        # Add the svc preview widget for input
        self.svc_preview = SVCpreview(filename, 0)
        self.collapsible_widget_result.add_widget(self.svc_preview)

        # Generate Synthetic Data button
        button_layout = QtWidgets.QVBoxLayout()
        self.generate_data_button = QtWidgets.QPushButton("Generate Synthetic Data", self)
        self.generate_data_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 14px; 
                font-weight: 600; 
                padding: 10px 20px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
            """
        )
        self.generate_data_button.setFixedSize(250, 50)
        self.generate_data_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor)) # put the button at the bottom
        self.generate_data_button.clicked.connect(self.on_generate_data)

        button_layout.addWidget(self.generate_data_button, alignment=QtCore.Qt.AlignCenter)

        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        button_layout.addItem(spacer)

        # Adding the button to the main layout
        scroll_layout.addLayout(button_layout)

        # Automatically open file preview widget after 2 secs
        QtCore.QTimer.singleShot(2000, lambda: self.collapsible_widget_file_preview.toggle_container(True))
        
    def on_generate_data(self):
        """Generate synthetic data and update the UI"""
        num_augmented_files = self.spin_box_widget.value()  # Get number of augmented files

        # Load the pre-trained VAE model
        vae_model = load_pretrained_vae('../model/pentab_vae_models/final_vae_model.h5')

        # Use the dynamic directory for file access
        svc_directory = self.local_page.current_directory

        # Start model training thread
        self.model_thread = ModelTrainingThread(
            vae_model, processed_data, data_frames,
            num_augmented_files, avg_data_points,
            scalers, original_data_frames, input_filenames
        )
        
        self.model_thread.log_signal.connect(self.process_log_widget.append)  # Update process log
        self.model_thread.finished.connect(self.on_training_finished)
        self.model_thread.start()

        self.generate_button.setEnabled(False)

    def on_training_finished(self):
        self.generate_button.setEnabled(True)

        # Full path to the zip file
        zip_path = os.path.join(self.local_page.current_directory, 'augmented_data.zip')

        self.output_widget.setText(f"Synthetic data generated. Download here: {zip_path}")
        self.process_log_widget.append(f"Synthetic data saved at {zip_path}")
        self.update_results_widget(zip_path)

    def update_results_widget(self, zip_path):
        """Display the contents of the zip file in the Results widget."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            for file in files:
                self.results_widget.addItem(f"File: {file}")


    def show_reset_confirmation_dialog(self):
        """Show a confirmation dialog before resetting the state."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Discard and Retry")
        message_box.setText("Are you sure you want to discard your current handwriting and start over?")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        message_box.setDefaultButton(QtWidgets.QMessageBox.No)
        
        message_box.setStyleSheet("QPushButton { font-size: 14px; }")
        
        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Yes:
            self.reset_state()

    def clear_all_drawings(self):
        """Clear all added files and reset the state."""
        self.file_list.clear()  # Empty the file list
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
    window.resize(800, 600)  # Adjust window size for the embedded browser
    window.show()
    sys.exit(app.exec_())