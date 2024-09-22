from PyQt5 import QtWidgets, QtGui, QtCore
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_container_widget import FileContainerWidget
from components.widget.slider_widget import SliderWidget
from components.button.DragDrop_Button import DragDrop_Button
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
            new_file_container = FileContainerWidget(file_name, self)
            new_file_container.hide_download_button()
            new_file_container.hide_retry_button()
            new_file_container.remove_file_signal.connect(self.file_upload_widget.remove_file)  # Connect remove signal
            self.file_container_layout.addWidget(new_file_container)
    
    def add_more_files(self):
        self.file_upload_widget.open_file_dialog()
        
if __name__ == "__main__":
    
    
    app = QtWidgets.QApplication([])
    window = Workplace()
    window.show()
    app.exec_()
