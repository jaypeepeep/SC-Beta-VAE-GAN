from PyQt5 import QtWidgets, QtCore, QtGui
import os

class FilePreviewWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(FilePreviewWidget, self).__init__(parent)
        self.uploaded_files = []
        self.setupUi()

    def setupUi(self):
        self.container_widget = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container_widget)

        # Horizontal layout to hold the file label and button on the same line
        self.header_layout = QtWidgets.QHBoxLayout()

        # Horizontal layout for first filename and select file button
        self.filename_button_layout = QtWidgets.QHBoxLayout()
        self.filename = QtWidgets.QLabel("Filename", self.container_widget)
        self.filename.setStyleSheet("font-family: Montserrat; font-size: 14px; font-weight: bold;")
        self.filename_button_layout.addWidget(self.filename, alignment=QtCore.Qt.AlignLeft)

        # Select file button
        self.select_file_button = QtWidgets.QPushButton("Select Files", self.container_widget)
        self.select_file_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 15px; 
                font-weight: 600; 
                padding: 5px 15px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
            """
        )
        self.select_file_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.select_file_button.clicked.connect(self.select_file)

        # Set fixed size for the button (CHANGED)
        self.select_file_button.setFixedSize(150, 40)  # Width: 150px, Height: 40px

        self.filename_button_layout.addWidget(self.select_file_button, alignment=QtCore.Qt.AlignRight)

        # Add the filename and button layout to the first text preview layout
        self.header_layout.addLayout(self.filename_button_layout)

        self.container_layout.addLayout(self.header_layout)

        # Text preview for the uploaded file
        self.text_preview = QtWidgets.QTextEdit(self.container_widget)
        self.text_preview.setReadOnly(True)
        self.text_preview.setFixedHeight(300)
        self.text_preview.setStyleSheet(
            """
            QTextEdit {
                background-color: white; 
                border: 1px solid #dcdcdc; 
                font-family: Montserrat; 
                font-size: 12px;
            }
            QTextEdit QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background: #003333; 
                min-height: 30px;
                border-radius: 4px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background: #005555;
            }
            QTextEdit QScrollBar::add-line:vertical, 
            QTextEdit QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
            }
            QTextEdit QScrollBar::add-page:vertical, 
            QTextEdit QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """
        )
        # Add the text preview to the container layout
        self.container_layout.addWidget(self.text_preview)

        # Set the layout for the widget
        self.setLayout(self.container_layout)

        # Automatically open the widget
        self.setVisible(True)


    def set_uploaded_files(self, files):
        """Set the list of uploaded files and display the first one."""
        self.uploaded_files = files
        if files:
            self.display_file_contents(files[0])
        else:
            self.clear()

    def display_file_contents(self, filename):
        """Read the contents of the file and display it in the text preview."""
        try:
            # Construct the uploads folder path and the full file path
            uploads_folder = os.path.join(os.path.dirname(__file__), "../../files/uploads")
            file_path = os.path.join(uploads_folder, filename)

            # Read the file and set its content to the text preview
            with open(file_path, "r") as file:
                content = file.read()
            self.filename.setText(os.path.basename(filename))
            self.text_preview.setPlainText(content)
        except Exception as e:
            self.text_preview.setPlainText(f"Error reading file: {str(e)}")

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
                padding: 5px 15px;
                border-radius: 5px;
                font-size: 10px;
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
            full_path = next(f for f in self.uploaded_files if os.path.basename(f) == selected_file)
            self.display_file_contents(full_path)

    def setText(self, text):
        """Method to set text in the text preview."""
        self.text_preview.setPlainText(text)

    def clear(self):
        """Clear the file preview widget."""
        self.filename.setText("Filename")
        self.text_preview.clear()
        self.uploaded_files.clear()