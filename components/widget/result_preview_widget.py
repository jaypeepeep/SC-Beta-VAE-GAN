from PyQt5 import QtWidgets, QtCore, QtGui
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SVCpreview(QtWidgets.QWidget):
    def __init__(self, input=None, output=None, metrics=None, parent=None):
        super(SVCpreview, self).__init__(parent)
        self.setupUi()
        self.uploaded_files = []
        self.augmented_files = []
        self.original_absolute_files = []

        if input:
            self.display_file_contents(input, 0)  # Display content in the first text preview
            self.display_graph_contents(input, 0)
        if output:
            self.display_file_contents(output, 1)  # Display content in the second text preview
            self.display_graph_contents(output, 1)
        if metrics:
            self.display_metrics(metrics)

    def setupUi(self):
        self.container_widget = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container_widget)

        # Horizontal layout for text previews and their labels
        self.preview_layout = QtWidgets.QHBoxLayout()

        # Vertical layout for first text preview
        self.text_preview1_layout = QtWidgets.QVBoxLayout()
        self.label1 = QtWidgets.QLabel("Input", self.container_widget)
        self.label1.setStyleSheet("font-family: Montserrat; font-size: 14px; font-weight: bold; text-align: center;")
        self.text_preview1_layout.addWidget(self.label1, alignment=QtCore.Qt.AlignCenter)

        # Horizontal layout for first filename and select file button
        self.filename_button_layout1 = QtWidgets.QHBoxLayout()
        self.filename1 = QtWidgets.QLabel("Filename", self.container_widget)
        self.filename1.setStyleSheet("font-family: Montserrat; font-size: 14px; font-weight: bold;")
        self.filename_button_layout1.addWidget(self.filename1, alignment=QtCore.Qt.AlignLeft)

        # Select file button
        self.select_file_button1 = QtWidgets.QPushButton("Select Files", self.container_widget)
        self.select_file_button1.setStyleSheet(
            """
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
            """
        )
        self.select_file_button1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.select_file_button1.clicked.connect(self.select_file)
        self.filename_button_layout1.addWidget(self.select_file_button1, alignment=QtCore.Qt.AlignRight)

        # Add the filename and button layout to the first text preview layout
        self.text_preview1_layout.addLayout(self.filename_button_layout1)

        # Text preview for the first file
        self.text_preview1 = QtWidgets.QTextEdit(self.container_widget)
        self.text_preview1.setReadOnly(True)
        self.text_preview1.setFixedHeight(300)
        self.text_preview1.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )
        self.text_preview1_layout.addWidget(self.text_preview1)

        # Graph container for input
        self.input_graph_container = QtWidgets.QWidget(self.container_widget)
        self.input_graph_container.setFixedHeight(400)
        self.input_graph_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dcdcdc;")
        self.input_graph_layout = QtWidgets.QVBoxLayout(self.input_graph_container)
        self.text_preview1_layout.addWidget(self.input_graph_container)

        # Vertical layout for second text preview
        self.text_preview2_layout = QtWidgets.QVBoxLayout()
        self.label2 = QtWidgets.QLabel("Output", self.container_widget)
        self.label2.setStyleSheet("font-family: Montserrat; font-size: 14px; font-weight: bold; text-align: center;")
        self.text_preview2_layout.addWidget(self.label2, alignment=QtCore.Qt.AlignCenter)

        # Horizontal layout for second filename and select file button
        self.filename_button_layout2 = QtWidgets.QHBoxLayout()
        self.filename2 = QtWidgets.QLabel("Filename", self.container_widget)
        self.filename2.setStyleSheet("font-family: Montserrat; font-size: 14px; font-weight: bold;")
        self.filename_button_layout2.addWidget(self.filename2, alignment=QtCore.Qt.AlignLeft)

        # Select file button
        self.select_file_button2 = QtWidgets.QPushButton("Select Files", self.container_widget)
        self.select_file_button2.setStyleSheet(
            """
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
            """
        )
        self.select_file_button2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.select_file_button2.clicked.connect(self.select_augmented_file)
        self.filename_button_layout2.addWidget(self.select_file_button2, alignment=QtCore.Qt.AlignRight)

        # Add the filename and button layout to the second text preview layout
        self.text_preview2_layout.addLayout(self.filename_button_layout2)

        # Text preview for the second file
        self.text_preview2 = QtWidgets.QTextEdit(self.container_widget)
        self.text_preview2.setReadOnly(True)
        self.text_preview2.setFixedHeight(300)
        self.text_preview2.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )
        self.text_preview2_layout.addWidget(self.text_preview2)

        self.output_graph_container = QtWidgets.QWidget(self.container_widget)
        self.output_graph_container.setFixedHeight(400)
        self.output_graph_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dcdcdc;")
        self.output_graph_layout = QtWidgets.QVBoxLayout(self.output_graph_container)
        self.text_preview2_layout.addWidget(self.output_graph_container)

        # Add both vertical layouts to the horizontal layout
        self.preview_layout.addLayout(self.text_preview1_layout)
        self.preview_layout.addLayout(self.text_preview2_layout)

        # Add the horizontal layout to the container layout
        self.container_layout.addLayout(self.preview_layout)

        # Results text area
        self.results_text = QtWidgets.QTextEdit(self.container_widget)
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(225)
        self.results_text.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 14px;"
        )
        self.container_layout.addWidget(self.results_text)

        self.results_text.setPlainText("Results")

        # Set the layout for the widget
        self.setLayout(self.container_layout)

    def add_graph_containers(self):
        # Graph container for output
        self.second_output_graph_container = QtWidgets.QWidget(self.container_widget)
        self.second_output_graph_container.setFixedHeight(500)
        self.second_output_graph_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dcdcdc;")
        self.second_output_graph_layout = QtWidgets.QVBoxLayout(self.second_output_graph_container)
        self.text_preview2_layout.addWidget(self.second_output_graph_container)

        # Second graph container for input
        self.second_input_graph_container = QtWidgets.QWidget(self.container_widget)
        self.second_input_graph_container.setFixedHeight(500)
        self.second_input_graph_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dcdcdc;")
        self.second_input_graph_layout = QtWidgets.QVBoxLayout(self.second_input_graph_container)
        self.text_preview1_layout.addWidget(self.second_input_graph_container)

    def remove_graph_containers(self):
        # Remove and delete the second output graph container
        if hasattr(self, 'second_output_graph_container'):
            self.text_preview2_layout.removeWidget(self.second_output_graph_container)
            self.second_output_graph_container.deleteLater()
            del self.second_output_graph_container
            del self.second_output_graph_layout

        # Remove and delete the second input graph container
        if hasattr(self, 'second_input_graph_container'):
            self.text_preview1_layout.removeWidget(self.second_input_graph_container)
            self.second_input_graph_container.deleteLater()
            del self.second_input_graph_container
            del self.second_input_graph_layout

    def display_handwriting_contents(self, file_path, preview_index):
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            df.columns = ['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']
            
            # Separate strokes based on pen status
            on_surface = df[df['pen_status'] == 1]
            in_air = df[df['pen_status'] == 0]

            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(on_surface['x'], -on_surface['y'], c='b', s=1, alpha=0.7, label='On Surface')
            ax.scatter(in_air['x'], -in_air['y'], c='r', s=1, alpha=0.7, label='In Air')
            ax.set_title(f'Time Series Handwriting Visualization for {os.path.basename(file_path)}')
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.legend()

            ax.invert_yaxis() 

            canvas = FigureCanvas(fig)
            if preview_index == 0:
                self.clear_layout(self.second_input_graph_layout)
                self.second_input_graph_layout.addWidget(canvas)
                canvas.draw() 
            else:
                self.clear_layout(self.second_output_graph_layout)
                self.second_output_graph_layout.addWidget(canvas)
                canvas.draw() 

        except:
            pass

    def display_emothaw_contents(self, file_path, preview_index):
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            df.columns = ['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']
            
            # Separate strokes based on pen status
            on_surface = df[df['pen_status'] == 1]
            in_air = df[df['pen_status'] == 0]

            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(on_surface['y'], on_surface['x'], c='b', s=1, alpha=0.7, label='On Surface')
            ax.scatter(in_air['y'], in_air['x'], c='r', s=1, alpha=0.7, label='In Air')
            ax.set_title(f'Time Series Handwriting Visualization for {os.path.basename(file_path)}')
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.legend()

            ax.set_xlim(ax.get_xlim()[::-1])  # Reverse the x-axis limits
            ax.set_ylim(ax.get_ylim()[::-1])  # Reverse the y-axis limits


            canvas = FigureCanvas(fig)
            if preview_index == 0:
                self.clear_layout(self.second_input_graph_layout)
                self.second_input_graph_layout.addWidget(canvas)
                canvas.draw() 
            else:
                self.clear_layout(self.second_output_graph_layout)
                self.second_output_graph_layout.addWidget(canvas)
                canvas.draw() 

        except:
            pass

    def add_result_text(self, text):
  
        # Get the current text
        current_text = self.results_text.toPlainText()
        
        # If there's already text, add a newline before the new text
        if current_text:
            new_text = current_text + "\n" + text
        else:
            new_text = text
        
        # Set the updated text
        self.results_text.setPlainText(new_text)
        

    def display_file_contents(self, filename, preview_index):
        """Read the contents of the file and display it in the appropriate text preview."""
        try:
            # Ensure the file path is absolute
            if not os.path.isabs(filename):
                filename = os.path.abspath(filename)

            # Check if the file exists before attempting to read
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")

            # Read and display the content
            with open(filename, "r") as file:
                content = file.read()
            if preview_index == 0:
                self.filename1.setText(os.path.basename(filename))
                self.text_preview1.setPlainText(content)
            else:
                self.filename2.setText(os.path.basename(filename))
                self.text_preview2.setPlainText(content)
        except Exception as e:
            error_message = f"Error reading file: {str(e)}"
            if preview_index == 0:
                self.text_preview1.setPlainText(error_message)
            else:
                self.text_preview2.setPlainText(error_message)

    def display_graph_contents(self, filename, preview_index):
        """Read the contents of the file and display it in the appropriate graph preview."""
        try:
            # Load the .svc file into a pandas DataFrame
            columns = ['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']
            data = pd.read_csv(filename, sep=' ', names=columns, header=None)

            for col in columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna()  # Remove rows with any NaN values


            # Remove 'pen_status' column from the data
            data = data.drop(columns=['pen_status'])

            # Create a mask for gaps in the timestamp greater than 8 units
            gap_threshold = 8
            timestamp_diff = data['timestamp'].diff()  # Get the differences between consecutive timestamps
            gap_mask = timestamp_diff > gap_threshold  # Find where gaps are greater than 8 units

            # Apply the mask by setting values to NaN where the gap occurs
            for col in columns:
                if col != 'timestamp':
                    data.loc[gap_mask, col] = np.nan  # Set the column values to NaN where there's a gap

            # Re-load 'pen_status' for handling color change logic
            pen_status = pd.read_csv(filename, sep=' ', names=['pen_status'], header=None, usecols=[3])

            # Function to change color based on pen status, unique color per variable
            def plot_segmented_lines(ax, x, y, status, color_down, color_up):
                start_idx = 0
                while start_idx < len(status):
                    pen_down = status[start_idx] == 1
                    segment_color = color_down if pen_down else color_up

                    try:
                        next_idx = next(i for i in range(start_idx + 1, len(status)) if status[i] != status[start_idx])
                    except StopIteration:
                        next_idx = len(status)

                    ax.plot(x[start_idx:next_idx], y[start_idx:next_idx], color=segment_color, label='_nolegend_')
                    start_idx = next_idx

            # Define distinct colors for each column when pen is down and a common color for pen up
            colors_down = {
                'x': 'blue',
                'y': 'green',
                'pressure': 'orange',
                'azimuth': 'red',
                'altitude': 'purple'
            }

            color_up = 'Cyan'  # Common pen up color for all variables

            # Create a new figure for the plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot each column with changing color based on pen status
            for col in ['x', 'y', 'pressure', 'azimuth', 'altitude']:
                color_down = colors_down[col]
                plot_segmented_lines(ax, data['timestamp'], data[col], pen_status['pen_status'], color_down, color_up)

            # Add titles and labels
            plt.title(f'Time Series Plot for {os.path.basename(filename)}')
            plt.xlabel('Timestamp')
            plt.ylabel('Values')

            # Create a legend manually with reduced font size
            for col in ['x', 'y', 'pressure', 'azimuth', 'altitude']:
                ax.plot([], [], color=colors_down[col], label=f'{col} (pen down)')
            ax.plot([], [], color=color_up, label='(pen up)')

            plt.legend(loc='upper right', fontsize='small')  # Use 'small' or a specific size like 8

            if preview_index == 0:
                self.clear_layout(self.input_graph_layout)
                canvas = FigureCanvas(fig)
                self.input_graph_layout.addWidget(canvas)
                canvas.draw()
            else:
                self.clear_layout(self.output_graph_layout)
                canvas = FigureCanvas(fig)
                self.output_graph_layout.addWidget(canvas)
                canvas.draw()

        except Exception as e:
            error_message = f"Error reading or displaying graph: {str(e)}"
            if preview_index == 0:
                self.text_preview1.setPlainText(error_message)
            else:
                self.text_preview2.setPlainText(error_message)
    
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def setText(self, text1, text2, results_text):
        """Method to set text in both text previews and the results area."""
        self.text_preview1.setPlainText(text1)
        self.text_preview2.setPlainText(text2)
        self.results_text.setPlainText(results_text)

    def set_uploaded_files(self, files):
        """
        Set the list of uploaded files and display the first one.
        Ensure paths are converted to absolute paths if they are not already.
        """
        # Convert any relative paths to absolute paths explicitly pointing to the uploads directory
        self.uploaded_files = [os.path.abspath(os.path.join('uploads', os.path.basename(file))) for file in files]
        
        # Display the first file if available
        if self.uploaded_files:
            try:
                first_file = self.uploaded_files[0]
                self.display_file_contents(first_file, 0)
                self.display_graph_contents(first_file, 0)
                self.display_handwriting_contents(first_file, 0)
            except Exception as e:
                print(f"Error displaying the first uploaded file: {e}")


            
    def select_file(self):
        """Open a custom dialog to select a file from the uploaded files, showing only the file name."""
        if not self.uploaded_files:
            QtWidgets.QMessageBox.warning(self, "No Files", "No files have been uploaded yet.")
            return

        # Display only the file names, not the paths
        file_names = [os.path.basename(file) for file in self.uploaded_files]

        # Create a custom dialog box
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select File")
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #f0f0f0; 
                font-family: Montserrat;
                padding: 20px;              
                border-radius: 10px;        
            }
            QPushButton {
                background-color: #003333;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005555;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #dcdcdc;
                padding: 5px;
                margin: 5px 0;
                font-size: 12px;
            }
            """
        )

        # Create a vertical layout for the dialog
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(10)

        # Add a label to guide the user
        label = QtWidgets.QLabel("Choose a file to preview:", dialog)
        label.setStyleSheet(
            """
            QLabel {
                font-size: 16px; 
                font-weight: bold; 
                color: black;      
                background: none;  
                padding: 0;      
                margin-bottom: 5px;  
            }
            """
        )
        layout.addWidget(label)

        # Create a list widget to display file names
        list_widget = QtWidgets.QListWidget(dialog)
        list_widget.addItems(file_names)
        
        # Increase the list height to match the dialog size
        list_widget.setFixedHeight(150)  # Set a fixed height for the list
        
        layout.addWidget(list_widget)

        # Create a horizontal layout for buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Create a 'Cancel' button
        cancel_button = QtWidgets.QPushButton("Cancel", dialog)
        cancel_button.clicked.connect(dialog.reject)  # Reject the dialog on cancel
        button_layout.addWidget(cancel_button)

        # Create a 'Select' button
        select_button = QtWidgets.QPushButton("Select", dialog)
        select_button.clicked.connect(dialog.accept)  # Accept the dialog on select
        button_layout.addWidget(select_button)

        # Add button layout to the main layout
        layout.addLayout(button_layout)

        # Execute the dialog and get the selected file
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_file = list_widget.currentItem().text()
            # Find the full path of the selected file
            full_path = next(f for f in self.original_absolute_files if os.path.basename(f) == selected_file)
            self.display_file_contents(full_path, 0)
            self.display_graph_contents(full_path, 0)
            self.display_handwriting_contents(full_path, 0)

            QtCore.QTimer.singleShot(100, lambda: self.render_graph1(full_path))

    def render_graph1(self, full_path):
        self.display_graph_contents(full_path, 0)
        self.input_graph_container.update()
        self.input_graph_container.repaint()
        self.input_graph_container.layout().update()
        self.input_graph_container.setVisible(True)
        QtWidgets.QApplication.processEvents()

    def set_original_absolute_files(self, files):
        self.original_absolute_files = files

    def set_augmented_files(self, files):
        """Set the list of uploaded files and display the first one."""
        self.augmented_files = files
            
    def select_augmented_file(self):
        """Open a custom dialog to select a file from the uploaded files, showing only the file name."""
        if not self.augmented_files:
            QtWidgets.QMessageBox.warning(self, "No Files", "No files have been uploaded yet.")
            return

        # Display only the file names, not the paths
        file_names = [os.path.basename(file) for file in self.augmented_files]

        # Create a custom dialog box
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select File")
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #f0f0f0; 
                font-family: Montserrat;
                padding: 20px;              
                border-radius: 10px;        
            }
            QPushButton {
                background-color: #003333;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005555;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #dcdcdc;
                padding: 5px;
                margin: 5px 0;
                font-size: 12px;
            }
            """
        )

        # Create a vertical layout for the dialog
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(10)

        # Add a label to guide the user
        label = QtWidgets.QLabel("Choose a file to preview:", dialog)
        label.setStyleSheet(
            """
            QLabel {
                font-size: 16px; 
                font-weight: bold; 
                color: black;      
                background: none;  
                padding: 0;      
                margin-bottom: 5px;  
            }
            """
        )
        layout.addWidget(label)

        # Create a list widget to display file names
        list_widget = QtWidgets.QListWidget(dialog)
        list_widget.addItems(file_names)
        
        # Increase the list height to match the dialog size
        list_widget.setFixedHeight(150)  # Set a fixed height for the list
        
        layout.addWidget(list_widget)

        # Create a horizontal layout for buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Create a 'Cancel' button
        cancel_button = QtWidgets.QPushButton("Cancel", dialog)
        cancel_button.clicked.connect(dialog.reject)  # Reject the dialog on cancel
        button_layout.addWidget(cancel_button)

        # Create a 'Select' button
        select_button = QtWidgets.QPushButton("Select", dialog)
        select_button.clicked.connect(dialog.accept)  # Accept the dialog on select
        button_layout.addWidget(select_button)

        # Add button layout to the main layout
        layout.addLayout(button_layout)

        # Execute the dialog and get the selected file
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_file = list_widget.currentItem().text()
            # Find the full path of the selected file
            full_path = next(f for f in self.augmented_files if os.path.basename(f) == selected_file)
            self.display_file_contents(full_path, 1)
            self.display_graph_contents(full_path, 1)
            self.display_handwriting_contents(full_path, 1)

            QtCore.QTimer.singleShot(100, lambda: self.render_graph(full_path))

    def render_graph(self, full_path):
        self.display_graph_contents(full_path, 1)
        self.output_graph_container.update()
        self.output_graph_container.repaint()
        self.output_graph_container.layout().update()
        self.output_graph_container.setVisible(True)
        QtWidgets.QApplication.processEvents()

    def clear(self):
        """Clear all displays and reset file lists."""
        # Clear text previews
        self.text_preview1.clear()
        self.text_preview2.clear()
        self.results_text.clear()

        # Reset filenames
        self.filename1.setText("Filename")
        self.filename2.setText("Filename")

        # Clear graph layout
        self.clear_layout(self.input_graph_layout)
        self.clear_layout(self.output_graph_layout)

        self.remove_graph_containers()

        # Update the widget
        self.update()


    @QtCore.pyqtSlot(str)
    def set_zip_path(self, zip_path):
        """Handle the ZIP file, list contents, and possibly display relevant file contents."""
        if not zipfile.is_zipfile(zip_path):
            self.results_text.setPlainText("Error: Invalid ZIP file.")
            return
        
        try:
            # Extract the zip contents to a temporary directory
            temp_dir = os.path.join(os.getcwd(), 'temp_extracted')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)  # Extract all files in the zip to temp directory
                file_list = zipf.namelist()

                # Make sure at least one file exists
                if len(file_list) > 0:
                    file1_path = os.path.join(temp_dir, file_list[0])
                    with open(file1_path, 'r') as file1:
                        content1 = file1.read()
                        self.filename1.setText(file_list[0])
                        self.text_preview1.setPlainText(content1)

                # If there is a second file, display it in the second preview
                if len(file_list) > 1:
                    file2_path = os.path.join(temp_dir, file_list[1])
                    with open(file2_path, 'r') as file2:
                        content2 = file2.read()
                        self.filename2.setText(file_list[1])
                        self.text_preview2.setPlainText(content2)
        except Exception as e:
            self.results_text.setPlainText(f"Error reading ZIP file: {str(e)}")