import os
from PyQt5 import QtWidgets, QtGui, QtCore


class Local(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Local, self).__init__(parent)
        self.current_directory = "./uploads"  # Initial directory path
        self.setupUi()
        self.load_files(self.current_directory)  # Load files from the initial directory

    def setupUi(self):
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Horizontal layout for Local Storage label and Change Location button
        header_layout = QtWidgets.QHBoxLayout()

        # Local Storage label
        self.local_storage_label = QtWidgets.QLabel("Local Storage", self)
        font = QtGui.QFont()
        font.setPointSize(14)  # Set font size to 14
        self.local_storage_label.setFont(font)
        self.local_storage_label.setAlignment(QtCore.Qt.AlignLeft)
        header_layout.addWidget(self.local_storage_label)

        # Spacer to push the button to the right
        header_layout.addStretch()

        # Change Location button
        self.change_location_button = QtWidgets.QPushButton("Change Location", self)
        self.change_location_button.setStyleSheet("border: 1px solid black; padding: 5px;")
        self.change_location_button.clicked.connect(self.change_directory)
        header_layout.addWidget(self.change_location_button)

        # Add header layout to main layout
        main_layout.addLayout(header_layout)

        # Path label
        self.path_label = QtWidgets.QLabel(self.current_directory, self)  # Initialize with the current directory path
        font.setPointSize(14)  # Set font size to 14
        self.path_label.setFont(font)
        self.path_label.setAlignment(QtCore.Qt.AlignLeft)
        main_layout.addWidget(self.path_label)

        # Files label
        self.files_label = QtWidgets.QLabel("Files", self)
        font.setPointSize(16)
        self.files_label.setFont(font)
        self.files_label.setAlignment(QtCore.Qt.AlignLeft)
        main_layout.addWidget(self.files_label)

        # Table for files
        self.table_widget = QtWidgets.QTableWidget(self)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Name", "Date"])
        self.table_widget.horizontalHeader().setStretchLastSection(True)  # Stretch last column to fit
        self.table_widget.setRowCount(0)  # Empty table for now
        main_layout.addWidget(self.table_widget)

        # Set layout margins
        main_layout.setContentsMargins(20, 20, 20, 20)

    def load_files(self, directory):
        """Populate the table with files and folders from the given directory"""
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Get list of files and directories
        files = os.listdir(directory)

        # Set number of rows in the table based on the number of files
        self.table_widget.setRowCount(len(files))

        for index, file_name in enumerate(files):
            # Get full path
            full_path = os.path.join(directory, file_name)

            # Set file/folder name
            name_item = QtWidgets.QTableWidgetItem(file_name)
            self.table_widget.setItem(index, 0, name_item)

            # Get the modification date
            mod_time = os.path.getmtime(full_path)
            mod_date = QtCore.QDateTime.fromSecsSinceEpoch(int(mod_time)).toString(QtCore.Qt.DefaultLocaleShortDate)

            # Set modification date
            date_item = QtWidgets.QTableWidgetItem(mod_date)
            self.table_widget.setItem(index, 1, date_item)

    def change_directory(self):
        """Open a directory picker and load files from the selected directory"""
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", self.current_directory)
        if new_dir:
            self.current_directory = new_dir  # Update current directory
            self.path_label.setText(self.current_directory)  # Update the path label
            self.load_files(self.current_directory)  # Reload files from the new directory


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    local_widget = Local()
    local_widget.show()
    sys.exit(app.exec_())
