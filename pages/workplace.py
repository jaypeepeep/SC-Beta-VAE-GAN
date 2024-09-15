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
        self.label_4 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(20)
        
        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        self.gridLayout.addWidget(self.collapsible_widget, 1, 0, 1, 1)
        # Add the "Add More Files" button
        self.add_file_button = QtWidgets.QPushButton("Add More Files", self)
        self.add_file_button.setStyleSheet(
            "border-radius: 5px; background-color: #535353; color: white; padding: 8px 16px;color: black; color: white; font-family: Montserrat; font-size: 14px; font-style: normal; font-weight: 600; line-height: normal;"
        )
        self.add_file_button.setFont(font)
        self.add_file_button.clicked.connect(self.add_more_files)  # Optional: connect to a function to add files
        self.collapsible_widget.add_widget(self.add_file_button)

        # Add a file container widget to the collapsible widget
        self.file_container = FileContainerWidget("example_file.txt", self)
        self.collapsible_widget.add_widget(self.file_container)
        
        # Ensure only the remove button is visible
        self.file_container.hide_download_button()
        self.file_container.hide_retry_button()

        # Add the slider widget directly to the layout
        self.slider_widget = SliderWidget(0, 10, self)
        self.collapsible_widget.add_widget(self.slider_widget)

    # Function to handle adding more file widgets (Optional)
    def add_more_files(self):
        # Example: add another file container widget when the button is clicked
        new_file_container = FileContainerWidget("new_file.txt", self)
        new_file_container.hide_download_button()
        new_file_container.hide_retry_button()
        self.collapsible_widget.add_widget(new_file_container)

