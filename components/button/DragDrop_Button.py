from PyQt5 import QtWidgets, QtGui, QtCore
import os
import shutil

class DragDrop_Button(QtWidgets.QWidget):
    file_uploaded = QtCore.pyqtSignal(list)
    def __init__(self, parent=None):
        super(DragDrop_Button, self).__init__(parent)
        self.uploaded_files = []
        self.setupUi()

    def setupUi(self):
        # Create layout for the widget
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Create drag and drop area container
        self.drop_area = QtWidgets.QWidget(self)
        self.drop_area.setStyleSheet(
            "border: 3px dashed #003333; background-color: transparent; padding: 20px; font-family: Montserrat; font-size: 14px; font-weight: bold; color: #535353;"
        )
        self.drop_area.setAcceptDrops(True)
        self.drop_area.setFixedHeight(250)

        # Create a layout for the drag and drop area
        self.drop_area_layout = QtWidgets.QVBoxLayout(self.drop_area)
        self.drop_area_layout.setAlignment(QtCore.Qt.AlignCenter)  
        self.drop_area_layout.setSpacing(1)  

        # Create the "Choose file" button
        self.file_button = QtWidgets.QPushButton("Choose File...", self.drop_area)
        self.file_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 12px; 
                font-weight: 600; 
                padding: 5px 15px; 
                border-radius: 5px; 
                border: none;
            }
            QPushButton:hover {
                background-color: #005555;  /* Change this to your desired hover color */
            }
            """
        )
        self.file_button.setFixedWidth(150)
        self.file_button.clicked.connect(self.open_file_dialog)

        # Create the "or" label (no border)
        self.or_label = QtWidgets.QLabel("or", self.drop_area)
        self.or_label.setAlignment(QtCore.Qt.AlignCenter)
        self.or_label.setStyleSheet(
            "font-family: Montserrat; font-size: 12px; font-weight: bold; color: #535353; border: none; padding: 5px;"
        )

        # Create the "Drop Files Here" label (no border)
        self.drop_label = QtWidgets.QLabel("Drop File Here", self.drop_area)
        self.drop_label.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "font-family: Montserrat; font-size: 12px; font-weight: bold; color: #535353; border: none; padding: 5px;"
        )

        # Add the button and labels to the drop area layout
        self.drop_area_layout.addWidget(self.file_button, 0, QtCore.Qt.AlignHCenter)
        self.drop_area_layout.addWidget(self.or_label, 0, QtCore.Qt.AlignHCenter)
        self.drop_area_layout.addWidget(self.drop_label, 0, QtCore.Qt.AlignHCenter)

        # Add the drop area to the main layout
        self.layout.addWidget(self.drop_area)

        # Connect drag and drop functionality
        self.drop_area.dragEnterEvent = self.drag_enter_event
        self.drop_area.dropEvent = self.drop_event

    def drag_enter_event(self, event):
        self.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event):
        file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
        # self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        if file_paths:
            self.handle_files(file_paths)

    def open_file_dialog(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select Files")
        if file_paths:
            self.handle_files(file_paths)

    def handle_files(self, file_paths):
        for file_path in file_paths:
            try:
                # Get the base name of the file (i.e., file name with extension)
                file_name = os.path.basename(file_path)
                
                # Construct the full path where the file will be saved
                destination_path = os.path.join(os.path.dirname(__file__), '../../files/uploads', file_name)
                
                # Copy the file to the save directory
                shutil.copy(file_path, destination_path)
                
                print(f"File '{file_name}' uploaded and saved to: {destination_path}")
            except Exception as e:
                print(f"Error saving file '{file_path}': {e}")

        # Emit the signal with the list of files after handling
        self.file_uploaded.emit(file_paths)

    def remove_file(self, file_path):
        if file_path in self.uploaded_files:
            self.uploaded_files.remove(file_path)
            if not self.uploaded_files:
                self.file_uploaded.emit(self.uploaded_files)

    def enterEvent(self, event):
        self.drop_area.setStyleSheet(
            "background-color: #A3BFBF; border: 3px dashed #003333;"
        )
        super(DragDrop_Button, self).enterEvent(event)

    def leaveEvent(self, event):
        self.drop_area.setStyleSheet(
            "border: 3px dashed #003333; background-color: transparent; padding: 20px; font-family: Montserrat; font-size: 14px; font-weight: bold; color: #535353;"
        )
        super(DragDrop_Button, self).leaveEvent(event)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DragDrop_Button()
    window.show()
    app.exec_()
