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
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setAlignment(QtCore.Qt.AlignTop)
        self.gridLayout.setContentsMargins(20, 10, 20, 20)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.setFont(font)

        # Create a layout for the labels (Local Storage and Path)
        label_layout = QtWidgets.QVBoxLayout()

        # Local Storage label
        self.local_storage_label = QtWidgets.QLabel("Local Storage", self)
        self.local_storage_label.setStyleSheet("""
            margin-left: 5px; 
            color: black;
            font-weight: bold;
            font-family: Montserrat; 
            font-size: 14px; 
        """)
        self.local_storage_label.setAlignment(QtCore.Qt.AlignLeft)
        label_layout.addWidget(self.local_storage_label)

        # Path label
        self.path_label = QtWidgets.QLabel(self.current_directory, self)  # Initialize with the current directory path
        self.path_label.setStyleSheet("""
            margin-left: 25px;
            color: black;
            font-family: Montserrat; 
            font-size: 14px;
            text-decoration: underline;
        """)
        self.path_label.setAlignment(QtCore.Qt.AlignLeft)
        label_layout.addWidget(self.path_label)

        # Horizontal layout for combining the label_layout and Change Location button
        header_layout = QtWidgets.QHBoxLayout()
        # Add the vertical labels layout to the horizontal layout
        header_layout.addLayout(label_layout)
        # Spacer to push the button to the right
        header_layout.addStretch()

        # Change Location button
        self.change_location_button = QtWidgets.QPushButton("Change Location", self)
        self.change_location_button.setStyleSheet("""
            QPushButton {
                margin-left: 10px;
                background-color: #003333;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 11px;
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
                line-height: 20px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
        """)
        self.change_location_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.change_location_button.clicked.connect(self.change_directory)
        header_layout.addWidget(self.change_location_button)

        # Add the header layout to the grid layout at row 0, spanning 2 columns
        self.gridLayout.addLayout(header_layout, 0, 0, 1, 2)

        # Files label
        self.files_label = QtWidgets.QLabel("Files", self)
        self.files_label.setStyleSheet("""
            margin-left: 5px; 
            color: black;
            font-family: Montserrat; 
            font-size: 14px; 
            font-weight: 600;
        """)
        self.files_label.setAlignment(QtCore.Qt.AlignLeft)
        self.gridLayout.addWidget(self.files_label, 1, 0, 1, 2)  # Span across 2 columns

        # # Scroll Area
        # self.scroll_area = QtWidgets.QScrollArea(self)
        # self.scroll_area.setWidgetResizable(True)
        # self.scroll_widget = QtWidgets.QWidget()
        # self.scroll_layout = QtWidgets.QGridLayout(self.scroll_widget)  # Grid layout for file items

        # # Add scroll area to the grid layout
        # self.scroll_area.setWidget(self.scroll_widget)
        # self.gridLayout.addWidget(self.scroll_area, 2, 0, 1, 2)
        
        # Table for files
        self.table_widget = QtWidgets.QTableWidget(self)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Name", "Date"])
        self.table_widget.horizontalHeader().setStretchLastSection(True)  # Stretch last column to fit
        self.table_widget.setRowCount(0)  # Empty table for now
        self.gridLayout.addWidget(self.table_widget)
        
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
