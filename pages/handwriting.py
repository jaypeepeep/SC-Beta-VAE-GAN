import subprocess
import requests
import os
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView
from components.button.handwriting_button import handwritingButton
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_preview_widget import FilePreviewWidget
from components.widget.process_log_widget import ProcessLogWidget
from components.widget.output_widget import OutputWidget
from components.widget.file_container_widget import FileContainerWidget 
from components.widget.plot_container_widget import PlotContainerWidget 
from components.widget.spin_box_widget import SpinBoxWidget
from components.widget.result_preview_widget import SVCpreview
import os
import sys
import requests

class Handwriting(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.drawing_done = False  # State to check if done button was clicked
        self.flask_process = None  # To keep track of the Flask process
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
            "Do you want to start drawing and handwriting?"
        )
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        message_box.setDefaultButton(QtWidgets.QMessageBox.Ok)

        # Apply stylesheet to customize button font size
        message_box.setStyleSheet("QPushButton { font-size: 14px; }")
    
        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Ok:
            self.run_flask_app()

    def run_flask_app(self):
        """Run the Flask app located in components/canvas/app.py and open it in the embedded browser."""
        flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../components/canvas/app.py'))
        
        # Run the Flask app as a subprocess
        self.flask_process = subprocess.Popen(['python', flask_app_path])

        # Display the embedded browser after a short delay to ensure Flask is running
        QtCore.QTimer.singleShot(5000, self.show_embedded_browser)

    def show_embedded_browser(self):
        """Show the Flask app inside the Handwriting page using QWebEngineView."""
        # Clear the current layout and show the embedded browser
        self.clear_layout()

        # Create a QWebEngineView and load the Flask app's URL
        self.webview = QWebEngineView(self)
        self.webview.setUrl(QtCore.QUrl("http://127.0.0.1:5000"))
        # Add webview to the layout
        self.layout.addWidget(self.webview)

        # Ensure the webview resizes responsively
        self.webview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Poll Flask to check if drawing is done and file is uploaded
        QtCore.QTimer.singleShot(5000, self.check_drawing_done)  # Adjust the delay if necessary

    def check_drawing_done(self):
        """Periodically check if the drawing is done by querying Flask."""
        try:
            response = requests.get("http://127.0.0.1:5000/check_upload")
            if response.status_code == 200:
                data = response.json()
                filename = data.get('filename')
                self.show_done_page(filename)  # Pass the filename to the next page
            else:
                print("File not uploaded yet, retrying...")
                QtCore.QTimer.singleShot(5000, self.check_drawing_done)  # Retry after delay
        except requests.ConnectionError:
            print("Flask server not ready, retrying...")
            QtCore.QTimer.singleShot(5000, self.check_drawing_done)  # Retry after delay if connection failed

    def show_done_page(self, filename):
        """Show the page after the drawing is completed."""
        self.clear_layout()

        # Create a scroll area to wrap the collapsible content
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # Create a widget that will be placed inside the scroll area
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setAlignment(QtCore.Qt.AlignTop)

        # Add the scroll area to the main layout
        scroll_area.setWidget(scroll_widget)
        self.layout.addWidget(scroll_area)

        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        scroll_layout.addWidget(self.collapsible_widget)
        self.collapsible_widget.toggle_container(True)

        # Add the plot container widget
        self.plot_container = PlotContainerWidget(self)
        self.plot_container.loadPlot(filename)
        self.collapsible_widget.add_widget(self.plot_container)

        # Add a file container widget to the collapsible widget with the actual filename
        self.file_container = FileContainerWidget(filename, self)
        self.collapsible_widget.add_widget(self.file_container)
        self.file_container.hide_remove_button()
        self.file_container.retry_button.clicked.connect(self.show_reset_confirmation_dialog)
        
        # Add the slider widget directly to the collapsible widget
        self.spin_box_widget =  SpinBoxWidget(0)
        self.collapsible_widget.add_widget(self.spin_box_widget)

        # Add the File Preview Widget
        self.collapsible_widget_result = CollapsibleWidget("File Preview", self)
        scroll_layout.addWidget(self.collapsible_widget_result)
        self.file_preview_widget = FilePreviewWidget(filename, self)
        self.collapsible_widget_result.add_widget(self.file_preview_widget)

        # Add the Process Log Widget
        self.collapsible_widget_result = CollapsibleWidget("Process Log", self)
        scroll_layout.addWidget(self.collapsible_widget_result)
        self.process_log_widget = ProcessLogWidget(self)
        self.collapsible_widget_result.add_widget(self.process_log_widget)

        # Add the Output Widget
        self.collapsible_widget_result = CollapsibleWidget("Output", self)
        scroll_layout.addWidget(self.collapsible_widget_result)
        self.output_widget = OutputWidget(self)
        self.collapsible_widget_result.add_widget(self.output_widget)

        # Call the collapsible widget component for result
        self.collapsible_widget_result = CollapsibleWidget("Result", self)
        scroll_layout.addWidget(self.collapsible_widget_result)

        # Add the svc preview widget for input
        self.svc_preview = SVCpreview(filename, 0)
        self.collapsible_widget_result.add_widget(self.svc_preview)


    def show_reset_confirmation_dialog(self):
        """Show a confirmation dialog before resetting the state."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Discard and Retry")
        message_box.setText("Are you sure you want to discard your current handwriting and start over?")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        message_box.setDefaultButton(QtWidgets.QMessageBox.No)
        
        message_box.setStyleSheet("QPushButton { font-size: 14px; }")
        
        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Yes:
            self.reset_state()

    def reset_state(self):
        """Reset the state and go back to the drawing page."""
        self.drawing_done = False
        self.show_drawing_page()

    def closeEvent(self, event):
        """Ensure the Flask app process is killed when the main window is closed."""
        if self.flask_process:
            self.flask_process.terminate()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.resize(800, 600)  # Adjust window size for the embedded browser
    window.show()
    sys.exit(app.exec_())
