from PyQt5 import QtWidgets, QtGui, QtCore
import os
import shutil
from font.dynamic_font_size import get_font_sizes  # Import your dynamic font sizes function

class DragDrop_Button(QtWidgets.QWidget):
    file_uploaded = QtCore.pyqtSignal(list)
    
    def __init__(self, parent=None):
        super(DragDrop_Button, self).__init__(parent)
        # Set up dynamic font family and sizes.
        self.font_family = "Montserrat"
        self.font_sizes = get_font_sizes()
        self.uploaded_files = []
        self.setupUi()

    def setupUi(self):
        # Create main layout for the widget.
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Create drag and drop area container.
        self.drop_area = QtWidgets.QWidget(self)
        self.drop_area.setStyleSheet(
            "border: 3px dashed #003333; background-color: transparent; padding: 20px; color: #535353;"
        )
        self.drop_area.setAcceptDrops(True)
        self.drop_area.setFixedHeight(250)

        # Create a layout for the drag and drop area.
        self.drop_area_layout = QtWidgets.QVBoxLayout(self.drop_area)
        self.drop_area_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_area_layout.setSpacing(1)

        # Create the "Choose File..." button.
        self.file_button = QtWidgets.QPushButton("Choose File...", self.drop_area)
        self.file_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                padding: 5px 15px; 
                border-radius: 5px; 
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005555;
            }
            """
        )
        # Set dynamic font for the button.
        file_button_font = QtGui.QFont(self.font_family, self.font_sizes["button"])
        self.file_button.setFont(file_button_font)
        self.file_button.clicked.connect(self.open_file_dialog)

        # Create the "or" label.
        self.or_label = QtWidgets.QLabel("or", self.drop_area)
        self.or_label.setAlignment(QtCore.Qt.AlignCenter)
        self.or_label.setStyleSheet(
            "color: #535353; border: none; padding: 5px;"
        )
        or_label_font = QtGui.QFont(self.font_family, self.font_sizes["content"])
        self.or_label.setFont(or_label_font)

        # Create the "Drop File Here" label.
        self.drop_label = QtWidgets.QLabel("Drop File Here", self.drop_area)
        self.drop_label.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "color: #535353; border: none; padding: 5px;"
        )
        drop_label_font = QtGui.QFont(self.font_family, self.font_sizes["content"])
        self.drop_label.setFont(drop_label_font)

        # Create the "Accepted files" label.
        self.accepted_files_label = QtWidgets.QLabel("Accepted files: *.svc", self.drop_area)
        self.accepted_files_label.setAlignment(QtCore.Qt.AlignCenter)
        self.accepted_files_label.setStyleSheet(
            "color: #535353; border: none; padding: 5px;"
        )
        accepted_files_font = QtGui.QFont(self.font_family, self.font_sizes["content"])
        self.accepted_files_label.setFont(accepted_files_font)

        # Add the button and labels to the drop area layout.
        self.drop_area_layout.addWidget(self.file_button, 0, QtCore.Qt.AlignHCenter)
        self.drop_area_layout.addWidget(self.or_label, 0, QtCore.Qt.AlignHCenter)
        self.drop_area_layout.addWidget(self.drop_label, 0, QtCore.Qt.AlignHCenter)
        self.drop_area_layout.addWidget(self.accepted_files_label, 0, QtCore.Qt.AlignHCenter)

        # Add the drop area to the main layout.
        self.layout.addWidget(self.drop_area)

        # Connect drag and drop functionality.
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
        if file_paths:
            self.handle_files(file_paths)

    def open_file_dialog(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select Files", filter="*.svc")  # Restrict to .svc files
        if file_paths:
            self.handle_files(file_paths)

    def handle_files(self, file_paths):
        valid_files = [file for file in file_paths if file.lower().endswith('.svc')]  # Filter out non-.svc files
        if not valid_files:
            QtWidgets.QMessageBox.warning(self, "Invalid File Type", "Please select only .svc files.")  # Show warning for invalid files
            return

        for file_path in valid_files:
            try:
                # Get the base name of the file.
                file_name = os.path.basename(file_path)
                # Construct the destination path.
                destination_path = os.path.join(os.path.dirname(__file__), '../../files/uploads', file_name)
                # Copy the file.
                shutil.copy(file_path, destination_path)
                print(f"File '{file_name}' uploaded and saved to: {destination_path}")
            except Exception as e:
                print(f"Error saving file '{file_path}': {e}")
        # Emit the signal with the list of files.
        self.file_uploaded.emit(valid_files)

    def remove_file(self, file_path):
        if file_path in self.uploaded_files:
            self.uploaded_files.remove(file_path)
            if not self.uploaded_files:
                self.file_uploaded.emit(self.uploaded_files)

    def enterEvent(self, event):
        # Update drop area style on hover.
        self.drop_area.setStyleSheet(
            "background-color: #A3BFBF; border: 3px dashed #003333; padding: 20px; color: #535353;"
        )
        super(DragDrop_Button, self).enterEvent(event)

    def leaveEvent(self, event):
        # Revert drop area style.
        self.drop_area.setStyleSheet(
            "border: 3px dashed #003333; background-color: transparent; padding: 20px; color: #535353;"
        )
        super(DragDrop_Button, self).leaveEvent(event)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = DragDrop_Button()
    window.show()
    sys.exit(app.exec_())
