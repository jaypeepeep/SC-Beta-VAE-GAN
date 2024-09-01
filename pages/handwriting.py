import webbrowser
import os
import sys
from PyQt5 import QtWidgets

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        # Create a layout
        layout = QtWidgets.QVBoxLayout(self)

        # Create the "Start Handwriting & Drawing" button
        self.draw_button = QtWidgets.QPushButton("Launch Canvas", self)
        layout.addWidget(self.draw_button)

        # Connect the button's clicked signal to the method that shows the confirmation dialog
        self.draw_button.clicked.connect(self.show_confirmation_dialog)

    def show_confirmation_dialog(self):
        # Show a message box to ask for confirmation
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Proceed to Handwriting & Drawing")
        message_box.setText(
            "To start handwriting and drawing, you'll be redirected to an external browser. Do you want to proceed?"
        )
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        message_box.setDefaultButton(QtWidgets.QMessageBox.Ok)

        # Show the message box and capture the user's response
        response = message_box.exec_()

        # If the user clicks "Ok", open the browser
        if response == QtWidgets.QMessageBox.Ok:
            self.open_browser()

    def open_browser(self):
        # Resolve the relative path to an absolute path
        html_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../components/canvas/Canvas.html'))

        # Convert the file path to a URL
        html_file_url = 'file://' + html_file_path

        # Open the HTML file in the default web browser
        webbrowser.open_new_tab(html_file_url)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.show()
    sys.exit(app.exec_())
