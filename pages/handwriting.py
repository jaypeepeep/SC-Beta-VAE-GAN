from PyQt5 import QtWidgets, QtCore, QtGui
from components.button.handwriting_button import handwritingButton
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_container_widget import FileContainerWidget 
from components.widget.plot_container_widget import PlotContainerWidget 
from components.widget.slider_widget import SliderWidget
import webbrowser
import os
import sys

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.drawing_done = False  # State to check if done button was clicked
        self.setupUi()

    def setupUi(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)  
        self.layout.setContentsMargins(50, 0, 50, 50)

        # Initial setup for the drawing page
        self.show_drawing_page()

    def clear_layout(self):
        """Clear the current layout and any child layouts."""
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            if item.layout():
                self.clear_layout_recursively(item.layout())
            del item

    def clear_layout_recursively(self, layout):
        """Recursively clear all widgets and child layouts in the given layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            if item.layout():
                self.clear_layout_recursively(item.layout())
            del item

    def show_drawing_page(self):
        """Show the drawing page layout."""
        self.clear_layout()

        # Create a layout for the text
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.setAlignment(QtCore.Qt.AlignCenter)
        top_layout.setContentsMargins(0, 20, 0, 20)

        # Add text
        top_text = QtWidgets.QLabel("Draw and Handwrite", self)
        top_text.setAlignment(QtCore.Qt.AlignCenter)
        top_text.setStyleSheet("font-size: 30px; font-weight: 300; color: #033;")
        top_layout.addWidget(top_text)

        # Create and add the handwriting button
        drawButton = handwritingButton(self)
        drawButton.setContentsMargins(50, 20, 50, 50)
        self.layout.addLayout(top_layout)
        self.layout.addWidget(drawButton)

        # Connect the button's click events
        drawButton.clicked.connect(self.show_confirmation_dialog)

    def show_confirmation_dialog(self):
        """Show a confirmation dialog before proceeding to the drawing page."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Proceed to Handwriting & Drawing")
        message_box.setText(
            "To start handwriting and drawing, you'll be redirected to an external browser. Do you want to proceed?"
        )
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        message_box.setDefaultButton(QtWidgets.QMessageBox.Ok)

        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Ok:
            self.open_browser()

    def open_browser(self):
        """Open the browser to display the handwriting canvas."""
        html_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../components/canvas/Canvas.html'))
        html_file_url = 'file://' + html_file_path
        webbrowser.open_new_tab(html_file_url)

        # Mark drawing as done and show the done page
        self.drawing_done = True
        self.show_done_page()

    def show_done_page(self):
        """Show the page after the drawing is completed."""
        self.clear_layout()
        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        self.layout.addWidget(self.collapsible_widget)

        # Add the plot container widget
        self.plot_container = PlotContainerWidget(self)
        self.collapsible_widget.add_widget(self.plot_container)

        # Add a file container widget to the collapsible widget
        self.file_container = FileContainerWidget("example_file.txt", self)
        self.collapsible_widget.add_widget(self.file_container)
        
        # Ensure only the remove button is visible
        self.file_container.hide_remove_button()
        self.file_container.retry_button.clicked.connect(self.reset_state)
        
        # Add the slider widget directly to the layout
        self.slider_widget = SliderWidget(0, 10, self)
        self.collapsible_widget.add_widget(self.slider_widget)

    def reset_state(self):
        """Reset the state and go back to the drawing page."""
        self.drawing_done = False
        self.show_drawing_page()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.resize(400, 400)  
    window.show()
    sys.exit(app.exec_())
