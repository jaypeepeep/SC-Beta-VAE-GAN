from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_container_widget import FileContainerWidget
from components.widget.slider_widget import SliderWidget
from components.button.DragDrop_Button import DragDrop_Button
from components.widget.result_svc_preview_widget import SVCpreview
import os

class Workplace(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Workplace, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setAlignment(QtCore.Qt.AlignTop)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.setFont(font)

        # Create a scroll area
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # Create a container widget for the scroll area content
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)

        # Set a size policy for the scroll widget that allows it to shrink
        self.scroll_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        # Add the scroll area to the main layout
        self.scroll_area.setWidget(self.scroll_widget)
        self.gridLayout.addWidget(self.scroll_area)

        # Call functions to set up collapsible components
        self.setup_input_collapsible()
        self.setup_preview_collapsible()
        self.setup_process_log_collapsible()
        self.setup_output_collapsible()
        self.setup_result_collapsible()

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
        self.gridLayout.addLayout(button_layout, 1, 0)
        
    def on_generate_data(self):
        # function when the button is clicked, right now, let's just put a print statement
        print("Synthetic data generated.")
    
    def setup_input_collapsible(self):
        """Set up the 'Input' collapsible widget and its contents."""
        font = QtGui.QFont()
        font.setPointSize(20)

        # Call the collapsible widget component for Input
        self.collapsible_widget_input = CollapsibleWidget("Input", self)
        self.scroll_layout.addWidget(self.collapsible_widget_input)

        # Add the FileUploadWidget
        self.file_upload_widget = DragDrop_Button(self)
        self.file_upload_widget.file_uploaded.connect(self.update_file_display)  # Connect the signal
        self.collapsible_widget_input.add_widget(self.file_upload_widget)

        # Add "Add More Files" button to Input collapsible widget
        self.add_file_button = QtWidgets.QPushButton("Add More Files", self)
        self.add_file_button.setStyleSheet(
            """
            QPushButton {
                background-color: #535353; 
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
        self.add_file_button.setFont(font)
        self.add_file_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.add_file_button.clicked.connect(self.add_more_files)
        self.collapsible_widget_input.add_widget(self.add_file_button)
        
        # Create a container to hold the file widgets and its layout
        self.file_container_widget = QtWidgets.QWidget(self)
        self.file_container_layout = QtWidgets.QVBoxLayout(self.file_container_widget)
        self.collapsible_widget_input.add_widget(self.file_container_widget)

        # File container widget for Input collapsible
        self.file_container = FileContainerWidget("example_file.txt", self)
        self.file_container_layout.addWidget(self.file_container)
        self.file_container.hide_download_button()
        self.file_container.hide_retry_button()
        # Slider widget in Input collapsible
        self.slider_widget = SliderWidget(0, 10, self)
        self.collapsible_widget_input.add_widget(self.slider_widget)

        # Initially hide other components
        self.file_upload_widget.setVisible(True)
        self.show_other_components(False)
        self.collapsible_widget_input.add_widget(self.slider_widget)

        # Open the collapsible widget by default
        self.collapsible_widget_input.toggle_container(True)

    
    def show_other_components(self, show=True):
        """Show or hide other components based on file upload."""
        self.add_file_button.setVisible(show)
        self.file_container_widget.setVisible(show)
        self.slider_widget.setVisible(show)

    def setup_preview_collapsible(self):
        """Set up the 'File Preview' collapsible widget and its contents."""
        # Call collapsible widget for File Preview
        self.collapsible_widget_preview = CollapsibleWidget("File Preview", self)
        self.scroll_layout.addWidget(self.collapsible_widget_preview)

        # Container for the content of the File Preview
        self.container_widget = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_widget.setStyleSheet(
            "background-color: #E0E0E0; border-radius: 5px; padding: 10px;"
        )

        # Horizontal layout to hold the file label and button on the same line
        self.header_layout = QtWidgets.QHBoxLayout()

        # Select file button
        self.select_file_button = QtWidgets.QPushButton("Select File", self.container_widget)
        self.select_file_button.setStyleSheet(
            "background-color: #003333; color: white; font-family: Montserrat; font-size: 14px; font-weight: 600; padding: 8px 16px; border-radius: 5px;"
        )
        self.header_layout.addWidget(self.select_file_button, alignment=QtCore.Qt.AlignRight)

        self.container_layout.addLayout(self.header_layout)

        # Text preview
        self.text_preview = QtWidgets.QTextEdit(self.container_widget)
        self.text_preview.setPlainText(
            "37128 37585 16837071 1 1800 670 49\n"
            "37128 37588 16837078 1 1800 670 141\n"
            "37128 37593 16837086 1 1800 670 174\n"
            "37121 37599 16837093 1 1800 680 218\n"
            "37111 37601 16837101 1 1800 680 268\n"
            "37098 37601 16837108 1 1800 680 286\n"
            "37079 37601 16837116 1 1800 680 310\n"
            "37055 37601 16837123 1 1800 680 332\n"
            "37025 37601 16837131 1 1800 680 338\n"
            "36992 37601 16837138 1 1800 680 347\n"
            "36957 37601 16837146 1 1800 680 358"
        )
        self.text_preview.setReadOnly(True)
        self.text_preview.setFixedHeight(150)
        self.text_preview.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )

        self.container_layout.addWidget(self.text_preview)
        self.collapsible_widget_preview.add_widget(self.container_widget)
    
    def setup_process_log_collapsible(self):
        """Set up the 'Process Log' collapsible widget and its contents, focusing on the text preview."""
        # Call collapsible widget for Process Log
        self.collapsible_widget_process_log = CollapsibleWidget("Process Log", self)
        self.scroll_layout.addWidget(self.collapsible_widget_process_log)

        # Container for the content of the Process Log
        self.container_widget_log = QtWidgets.QWidget(self)
        self.container_layout_log = QtWidgets.QVBoxLayout(self.container_widget_log)
        self.container_layout_log.setContentsMargins(10, 10, 10, 10)
        self.container_widget_log.setStyleSheet(
            "background-color: #E0E0E0; border-radius: 5px; padding: 10px;"
        )

        # Text preview for process log
        self.text_preview_log = QtWidgets.QTextEdit(self.container_widget_log)
        self.text_preview_log.setPlainText(
            "Epoch 1/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.91batch/s, loss=0.000568]\n"
            "Epoch 2/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000494]\n"
            "Epoch 3/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000807]\n"
            "Epoch 4/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000590]\n"
            "Epoch 5/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 6/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 7/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 8/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 9/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]"
        )
        self.text_preview_log.setReadOnly(True)
        self.text_preview_log.setFixedHeight(150)
        self.text_preview_log.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )

        # Add the text preview to the container layout
        self.container_layout_log.addWidget(self.text_preview_log)

        # Add the container widget to the collapsible widget
        self.collapsible_widget_process_log.add_widget(self.container_widget_log)

    def setup_output_collapsible(self):
        """Set up the 'Output' collapsible widget with file display and sample output images."""
        # Call the collapsible widget for Output
        self.collapsible_widget_output = CollapsibleWidget("Output", self)
        self.scroll_layout.addWidget(self.collapsible_widget_output)

        # File container widget for the file display section
        file_container = QtWidgets.QWidget(self)
        file_container.setStyleSheet("""
            background-color: #DEDEDE;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 0;
        """)

        # Shadow to the first file container
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(8)
        shadow_effect.setColor(QColor(0, 0, 0, 160))
        shadow_effect.setOffset(2)
        file_container.setGraphicsEffect(shadow_effect)

        file_container_layout = QtWidgets.QHBoxLayout(file_container)
        file_container_layout.setSpacing(20)
        file_container_layout.setContentsMargins(10, 10, 10, 10)

        # File name label
        file_label = QtWidgets.QLabel("Time-series_Data.zip")
        file_label.setStyleSheet("font-family: Montserrat; font-size: 14px; color: black;")
        file_container_layout.addWidget(file_label)

        # Container for buttons
        button_container = QtWidgets.QWidget(self)
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setSpacing(10)

        # Done Button
        done_button = QtWidgets.QPushButton("Done")
        done_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #003333;
                color: #003333;
                background-color: transparent;
                font-family: Montserrat;
                font-size: 14px;
                padding: 7px 20px;
                border-radius: 10px;
            }
        """)
        done_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(done_button)

        # Download button
        download_button = QtWidgets.QPushButton("Download")
        download_button.setStyleSheet("""
            QPushButton {
                background-color: #003333;
                color: white;
                font-family: Montserrat;
                font-size: 14px;
                padding: 7px 20px;
                border-radius: 10px;
            }
        """)
        download_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(download_button)

        # Delete button
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
        button_layout.addWidget(delete_button)

        # Button container added to the file container layout
        file_container_layout.addWidget(button_container, alignment=QtCore.Qt.AlignRight)

        # File container added to the collapsible widget
        self.collapsible_widget_output.add_widget(file_container)

        # Background for sample output
        images_outer_container = QtWidgets.QWidget(self)
        images_outer_container.setStyleSheet("background-color: #D9D9D9; border-radius: 0; padding: 15px;")
        images_outer_layout = QtWidgets.QVBoxLayout(images_outer_container)

        sample_output_label = QtWidgets.QLabel("Sample Output")
        sample_output_label.setStyleSheet("""
            font-family: Montserrat;
            font-size: 16px;
            color: black;
            font-weight: bold;
        """)
        sample_output_label.setAlignment(QtCore.Qt.AlignCenter)
        images_outer_layout.addWidget(sample_output_label)

        # Layout to hold OG and Augmented Images
        images_layout = QtWidgets.QHBoxLayout()

        # Original Input
        original_container = QtWidgets.QWidget(self)
        original_layout = QtWidgets.QVBoxLayout(original_container)
        original_container.setStyleSheet("background-color: white; padding: 5px; border-radius: 0;")

        original_label = QtWidgets.QLabel("Original Input")
        original_label.setAlignment(QtCore.Qt.AlignCenter)
        original_label.setStyleSheet("font-family: Montserrat; font-size: 12px; color: black;")

        original_image = QtWidgets.QLabel(self)
        original_image_path = self.get_image_path('original_data.png')
        original_qimage = QtGui.QImage(original_image_path)

        if original_qimage.isNull():
            original_image.setText("Image not found.")
        else:
            original_image.setPixmap(QtGui.QPixmap.fromImage(original_qimage).scaled(200, 200, QtCore.Qt.KeepAspectRatio))
        
        original_image.setAlignment(QtCore.Qt.AlignCenter)

        original_layout.addWidget(original_label)
        original_layout.addWidget(original_image)
        images_layout.addWidget(original_container)

        # Augmented Data
        augmented_container = QtWidgets.QWidget(self)
        augmented_layout = QtWidgets.QVBoxLayout(augmented_container)
        augmented_container.setStyleSheet("background-color: white; padding: 5px; border-radius: 0;")

        augmented_label = QtWidgets.QLabel("Augmented Data")
        augmented_label.setAlignment(QtCore.Qt.AlignCenter)
        augmented_label.setStyleSheet("font-family: Montserrat; font-size: 12px; color: black;")

        augmented_image = QtWidgets.QLabel(self)
        augmented_image_path = self.get_image_path('augmented_data.png')
        augmented_qimage = QtGui.QImage(augmented_image_path)

        if augmented_qimage.isNull():
            augmented_image.setText("Image not found.")
        else:
            augmented_image.setPixmap(QtGui.QPixmap.fromImage(augmented_qimage).scaled(200, 200, QtCore.Qt.KeepAspectRatio))
        
        augmented_image.setAlignment(QtCore.Qt.AlignCenter)

        augmented_layout.addWidget(augmented_label)
        augmented_layout.addWidget(augmented_image)
        images_layout.addWidget(augmented_container)

        # Images layout added to the outer container
        images_outer_layout.addLayout(images_layout)

        # Outer container added to the collapsible widget
        self.collapsible_widget_output.add_widget(images_outer_container)

    def setup_result_collapsible(self):
        """Set up the 'Result' collapsible widget and its contents."""

        # Call collapsible widget for Result
        self.collapsible_widget_result = CollapsibleWidget("Result", self)
        self.scroll_layout.addWidget(self.collapsible_widget_result)


    def update_file_display(self, uploaded_files):
        """Update the display of files based on uploaded files."""
        has_files = bool(uploaded_files)
        self.show_other_components(has_files)
        
        # Make the file upload widget visible if no files are uploaded
        self.file_upload_widget.setVisible(not has_files)
                
        # Clear existing widgets in the file container layout
        for i in reversed(range(self.file_container_layout.count())): 
            widget = self.file_container_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Re-add file containers for each uploaded file
        for file_path in uploaded_files:
            file_name = os.path.basename(file_path)
            self.svc_preview = SVCpreview(file_name, file_name)
            self.collapsible_widget_result.add_widget(self.svc_preview)
            new_file_container = FileContainerWidget(file_name, self)
            new_file_container.hide_download_button()
            new_file_container.hide_retry_button()
            new_file_container.remove_file_signal.connect(self.file_upload_widget.remove_file)  # Connect remove signal
            self.file_container_layout.addWidget(new_file_container)
    
    def add_more_files(self):
        self.file_upload_widget.open_file_dialog()
    
    def get_image_path(self, image_name):
        path = os.path.join(os.path.dirname(__file__), '..', 'icon', image_name)
        return path
        
if __name__ == "__main__":
    
    
    app = QtWidgets.QApplication([])
    window = Workplace()
    window.show()
    app.exec_()
