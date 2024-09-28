from PyQt5 import QtWidgets, QtCore
import os

class FilePreviewWidget(QtWidgets.QWidget):
    def __init__(self, filename=None, parent=None):
        super(FilePreviewWidget, self).__init__(parent)
        self.setupUi()
        if filename:
            self.display_file_contents(filename)  # Display content if filename is provided

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
            "background-color: #003333; color: white; font-family: Montserrat; font-size: 14px; font-weight: 600; padding: 8px 16px; border-radius: 5px;"
        )
        self.filename_button_layout.addWidget(self.select_file_button, alignment=QtCore.Qt.AlignRight)

        # Add the filename and button layout to the first text preview layout
        self.header_layout.addLayout(self.filename_button_layout)


        self.container_layout.addLayout(self.header_layout)

        # Text preview for the uploaded file
        self.text_preview = QtWidgets.QTextEdit(self.container_widget)
        self.text_preview.setReadOnly(True)
        self.text_preview.setFixedHeight(300)
        self.text_preview.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )

        # Add the text preview to the container layout
        self.container_layout.addWidget(self.text_preview)

        # Set the layout for the widget
        self.setLayout(self.container_layout)

        # Automatically open the widget
        self.setVisible(True)

    def display_file_contents(self, filename):
        """Read the contents of the file and display it in the text preview."""
        try:
            # Construct the uploads folder path and the full file path
            uploads_folder = os.path.join(os.path.dirname(__file__), "../../uploads")
            file_path = os.path.join(uploads_folder, filename)

            # Read the file and set its content to the text preview
            with open(file_path, "r") as file:
                content = file.read()
            self.filename.setText(os.path.basename(filename))
            self.text_preview.setPlainText(content)
        except Exception as e:
            self.text_preview.setPlainText(f"Error reading file: {str(e)}")

    def setText(self, text):
        """Method to set text in the text preview."""
        self.text_preview.setPlainText(text)
