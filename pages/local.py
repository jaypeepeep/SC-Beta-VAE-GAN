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
        font.setPointSize(8)
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
        header_layout.addLayout(label_layout)
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

        # Create and configure the table widget
        self.table_widget = QtWidgets.QTableWidget(self)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Name", "Date"])
        self.table_widget.horizontalHeader().setStretchLastSection(True)  # Stretch the last column to fit
        self.table_widget.setRowCount(0)  # Empty table for now

        # Set the width of the "Name" column to be wider
        self.table_widget.setColumnWidth(0, 600)  # Adjusted "Name" column width

        # Set font to size 14 for the table
        font_x = QtGui.QFont()
        font_x.setPointSize(8)
        self.table_widget.setFont(font_x)

        # Add hover effect for rows
        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: #f0f0f0;
            }
            QTableWidget::item:hover {
                background-color:#033;  
                color: white;
            }
            QTableWidget::item:selected {
                background-color: #033; 
                color: white;
            }
            QHeaderView::section {
                color: #033;
                padding: 5px;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }

            QTableWidget::item {
                padding: 10px;
                border: none;
            }

            QTableCornerButton::section {
                background-color: #f0f0f0;
                border: none;
            }
        """)

        # Add table to the layout
        self.gridLayout.addWidget(self.table_widget)

        # Connect the click event for preview
        self.table_widget.cellClicked.connect(self.preview_file)

    def load_files(self, directory):
        """Populate the table with files and folders from the given directory"""
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Get list of files and directories
        files = os.listdir(directory)

        # Set number of rows in the table based on the number of files
        self.table_widget.setRowCount(len(files))
        self.table_widget.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

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

    def preview_file(self, row, column):
        """Handle file preview when a file is clicked"""
        # Get the file name from the clicked row
        file_name = self.table_widget.item(row, 0).text()
        file_path = os.path.join(self.current_directory, file_name)

        # For now, let's handle basic previews:
        # 1. If it's an image file, show it in a QLabel
        # 2. Otherwise, print file path or open the file with the default program
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Preview image
            self.show_image_preview(file_path)
        else:
            # Non-image files - open in default program (or print path)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(file_path))

    def show_image_preview(self, file_path):
        """Display an image preview in a centered dialog with a close button, scaling the dialog to the image size."""
        # Load the image
        pixmap = QtGui.QPixmap(file_path)
        
        # Create a dialog for the image preview
        preview_dialog = QtWidgets.QDialog(self)
        preview_dialog.setWindowTitle("Image Preview")

        # Calculate the scaled size of the image while maintaining the aspect ratio
        scaled_pixmap = pixmap.scaled(600, 600, QtCore.Qt.KeepAspectRatio)
        dialog_width = scaled_pixmap.width() + 20  # Adding some padding
        dialog_height = scaled_pixmap.height() + 60  # Adding padding for the button

        # Set the dialog size based on the scaled image
        preview_dialog.setFixedSize(dialog_width, dialog_height)

        # Create a layout for the dialog
        layout = QtWidgets.QVBoxLayout(preview_dialog)

        # Add the image label to the layout
        image_label = QtWidgets.QLabel(preview_dialog)
        image_label.setPixmap(scaled_pixmap)
        layout.addWidget(image_label, 0, QtCore.Qt.AlignCenter)

        # Create a close button
        close_button = QtWidgets.QPushButton("Close", preview_dialog)
        close_button.clicked.connect(preview_dialog.close)
        close_button.setStyleSheet("""
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
        close_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Add the close button to the layout
        layout.addWidget(close_button, 0, QtCore.Qt.AlignCenter)

        # Center the dialog on the screen
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = preview_dialog.geometry()
        preview_dialog.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

        # Show the dialog
        preview_dialog.exec_()


    def change_directory(self):
        """Open a directory picker and load files from the selected directory"""
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", self.current_directory)
        if new_dir:
            self.current_directory = new_dir  # Update current directory
            self.path_label.setText(self.current_directory)  # Update the path label
            self.load_files(self.current_directory)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    local_widget = Local()
    local_widget.show()
    sys.exit(app.exec_())
