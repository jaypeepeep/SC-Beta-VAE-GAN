from PyQt5 import QtWidgets, QtCore
import os  # Import os for handling file paths


class SVCpreview(QtWidgets.QWidget):
    def __init__(self, filename1=None, filename2=None, parent=None):
        super(SVCpreview, self).__init__(parent)
        self.setupUi()
        if filename1:
            self.display_file_contents(
                filename1, 0
            )  # Display content in the first text preview
        if filename2:
            self.display_file_contents(
                filename2, 1
            )  # Display content in the second text preview

    def setupUi(self):
        self.container_widget = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container_widget)

        # Horizontal layout to hold the file label and button on the same line
        self.header_layout = QtWidgets.QHBoxLayout()

        # Select file button (temporarily just for layout)
        self.select_file_button = QtWidgets.QPushButton(
            "Select Files", self.container_widget
        )
        self.select_file_button.setStyleSheet(
            "background-color: #003333; color: white; font-family: Montserrat; font-size: 14px; font-weight: 600; padding: 8px 16px; border-radius: 5px;"
        )
        self.header_layout.addWidget(
            self.select_file_button, alignment=QtCore.Qt.AlignRight
        )

        self.container_layout.addLayout(self.header_layout)

        # Horizontal layout for two text previews
        self.preview_layout = QtWidgets.QHBoxLayout()

        # Text preview for the first file
        self.text_preview1 = QtWidgets.QTextEdit(self.container_widget)
        self.text_preview1.setReadOnly(True)
        self.text_preview1.setFixedHeight(150)
        self.text_preview1.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )

        # Text preview for the second file
        self.text_preview2 = QtWidgets.QTextEdit(self.container_widget)
        self.text_preview2.setReadOnly(True)
        self.text_preview2.setFixedHeight(150)
        self.text_preview2.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )

        # Add both text previews to the horizontal layout
        self.preview_layout.addWidget(self.text_preview1)
        self.preview_layout.addWidget(self.text_preview2)

        # Add the horizontal layout to the container layout
        self.container_layout.addLayout(self.preview_layout)

        # Set the layout for the widget
        self.setLayout(self.container_layout)

    def display_file_contents(self, filename, preview_index):
        """Read the contents of the file and display it in the appropriate text preview."""
        try:
            # Construct the uploads folder path and the full file path
            uploads_folder = os.path.join(os.path.dirname(__file__), "../../uploads")
            file_path = os.path.join(uploads_folder, filename)

            # Read the file and set its content to the appropriate text preview
            with open(file_path, "r") as file:
                content = file.read()
            if preview_index == 0:
                self.text_preview1.setPlainText(content)
            else:
                self.text_preview2.setPlainText(content)
        except Exception as e:
            if preview_index == 0:
                self.text_preview1.setPlainText(f"Error reading file: {str(e)}")
            else:
                self.text_preview2.setPlainText(f"Error reading file: {str(e)}")

    def setText(self, text1, text2):
        """Method to set text in both text previews."""
        self.text_preview1.setPlainText(text1)
        self.text_preview2.setPlainText(text2)
