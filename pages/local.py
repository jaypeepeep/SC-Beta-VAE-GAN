from PyQt5 import QtWidgets, QtGui, QtCore
from components.collapsible_widget import CollapsibleWidget
from components.file_container_widget import FileContainerWidget 
from components.slider_widget import SliderWidget
from components.plot_container_widget import PlotContainerWidget 

class Local(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Local, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)

        # Main label
        self.label_4 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setText("Local Page")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        self.gridLayout.addWidget(self.collapsible_widget, 1, 0, 1, 1)

        # Add the plot container widget
        self.plot_container = PlotContainerWidget(self)
        self.collapsible_widget.add_widget(self.plot_container)

        # Add a file container widget to the collapsible widget
        self.file_container = FileContainerWidget("example_file.txt", self)
        self.collapsible_widget.add_widget(self.file_container)
        
        # Ensure only the remove button is visible
        self.file_container.hide_download_button()
        self.file_container.hide_retry_button()

        # Add the "Add More Files" button
        self.add_file_button = QtWidgets.QPushButton("Add More Files", self)
        self.add_file_button.setStyleSheet(
            "border-radius: 5px; background-color: #535353; color: white; padding: 8px 16px;color: black; color: white; font-family: Montserrat; font-size: 20px; font-style: normal; font-weight: 600; line-height: normal;"
        )
        self.add_file_button.clicked.connect(self.add_more_files)  # Optional: connect to a function to add files
        self.collapsible_widget.add_widget(self.add_file_button)

                # Add the slider widget directly to the layout
        self.slider_widget = SliderWidget(0, 10, self)
        self.collapsible_widget.add_widget(self.slider_widget)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    local_widget = Local()
    local_widget.show()
    sys.exit(app.exec_())
