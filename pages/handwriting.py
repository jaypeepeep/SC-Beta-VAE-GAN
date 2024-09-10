import webbrowser
import os
import sys
from PyQt5 import QtWidgets, QtCore, QtGui

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.handwriting_button import handwritingButton

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)  
        layout.setContentsMargins(50, 0, 50, 50)  # Remove default margins to use spacers effectively

        # Create a layout for the text
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.setAlignment(QtCore.Qt.AlignCenter)  # Center items in top_layout
        top_layout.setContentsMargins(0, 20, 0, 20)  # Margins around image and text

        # Add text
        top_text = QtWidgets.QLabel("Draw and Handwrite", self)
        top_text.setAlignment(QtCore.Qt.AlignCenter)
        top_text.setStyleSheet("font-size: 30px; font-weight: 300; color: #033;")
        top_layout.addWidget(top_text)

        # Create and add the handwriting button
        self.drawButton = handwritingButton(self)
        self.drawButton.setContentsMargins(50, 20, 50, 50)  # Margins around the button

        # Add image, text, and button to the main layout
        layout.addLayout(top_layout)  # Add top layout with image and text
        layout.addWidget(self.drawButton)  # Add button

        # Connect the button's click events
        self.drawButton.clicked.connect(self.show_confirmation_dialog)


    def show_confirmation_dialog(self):
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
        html_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../components/canvas/Canvas.html'))
        html_file_url = 'file://' + html_file_path
        webbrowser.open_new_tab(html_file_url)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.resize(400, 400)  # Set initial size of the window
    window.show()
    sys.exit(app.exec_())
