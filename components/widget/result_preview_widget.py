from PyQt5 import QtWidgets, QtCore
import os

class SVCpreview(QtWidgets.QWidget):
    def __init__(self, input=None, output=None, parent=None):
        super(SVCpreview, self).__init__(parent)
        self.setupUi()
        if input:
            self.display_file_contents(input, 0)  # Display content in the first text preview
        if output:
            self.display_file_contents(output, 1)  # Display content in the second text preview

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
            "background-color: #003333; color: white; font-family: Montserrat; font-size: 14px; font-weight: 600; padding: 8px 16px; border-radius: 5px;"
        )
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
        self.input_graph_container.setFixedHeight(300)
        self.input_graph_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dcdcdc;")
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
            "background-color: #003333; color: white; font-family: Montserrat; font-size: 14px; font-weight: 600; padding: 8px 16px; border-radius: 5px;"
        )
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

        # Graph container for output
        self.output_graph_container = QtWidgets.QWidget(self.container_widget)
        self.output_graph_container.setFixedHeight(300)
        self.output_graph_container.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dcdcdc;")
        self.text_preview2_layout.addWidget(self.output_graph_container)

        # Add both vertical layouts to the horizontal layout
        self.preview_layout.addLayout(self.text_preview1_layout)
        self.preview_layout.addLayout(self.text_preview2_layout)

        # Add the horizontal layout to the container layout
        self.container_layout.addLayout(self.preview_layout)

        # Results text area
        self.results_text = QtWidgets.QTextEdit(self.container_widget)
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(100)
        self.results_text.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 14px;"
        )
        self.container_layout.addWidget(self.results_text)

        self.results_text.setPlainText("Results")

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

    def setText(self, text1, text2, results_text):
        """Method to set text in both text previews and the results area."""
        self.text_preview1.setPlainText(text1)
        self.text_preview2.setPlainText(text2)
        self.results_text.setPlainText(results_text)
