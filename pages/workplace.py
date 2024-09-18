from PyQt5 import QtWidgets, QtGui, QtCore
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_container_widget import FileContainerWidget 
from components.widget.slider_widget import SliderWidget
from components.widget.plot_container_widget import PlotContainerWidget

class Workplace(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Workplace, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setAlignment(QtCore.Qt.AlignTop)
        self.label_4 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(20)

        # Call the collapsible widget component
        self.collapsible_widget_input = CollapsibleWidget("Input", self)
        self.gridLayout.addWidget(self.collapsible_widget_input, 1, 0, 1, 1)

        # Add "Add More Files" button to Input collapsible widget
        self.add_file_button = QtWidgets.QPushButton("Add More Files", self)
        self.add_file_button.setStyleSheet(
            "border-radius: 5px; background-color: #535353; color: white; padding: 8px 16px; font-family: Montserrat; font-size: 14px; font-style: normal; font-weight: 600; line-height: normal;"
        )
        self.add_file_button.setFont(font)
        self.add_file_button.clicked.connect(self.add_more_files)
        self.collapsible_widget_input.add_widget(self.add_file_button)

        # File container widget for Input collapsible
        self.file_container = FileContainerWidget("example_file.txt", self)
        self.collapsible_widget_input.add_widget(self.file_container)
        self.file_container.hide_download_button()
        self.file_container.hide_retry_button()

        # Slider widget in Input collapsible
        self.slider_widget = SliderWidget(0, 10, self)
        self.collapsible_widget_input.add_widget(self.slider_widget)

        # Call collapsable widget for File Preview
        self.collapsible_widget_preview = CollapsibleWidget("File Preview", self)
        self.gridLayout.addWidget(self.collapsible_widget_preview, 2, 0, 1, 1)

        # Container for the content of the File Preview
        self.container_widget = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_widget.setStyleSheet(
            "background-color: #E0E0E0; border-radius: 5px; padding: 10px;"
        )

        # Horizontal layout to hold the file label and button on the same line
        self.header_layout = QtWidgets.QHBoxLayout()
        
        # Label for the file name
        self.file_label = QtWidgets.QLabel("Time-series_Data.svc", self.container_widget)
        self.file_label.setStyleSheet("font-size: 16px; font-family: Montserrat;")
        self.header_layout.addWidget(self.file_label, alignment=QtCore.Qt.AlignLeft)

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

    # Function to handle adding more file widgets (Optional)
    def add_more_files(self):
        new_file_container = FileContainerWidget("new_file.txt", self)
        new_file_container.hide_download_button()
        new_file_container.hide_retry_button()
        self.collapsible_widget_input.add_widget(new_file_container)

