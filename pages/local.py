import os
from PyQt5 import QtWidgets, QtGui, QtCore


class Local(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Local, self).__init__(parent)
        self.current_directory = "./uploads"  # Initial directory path
        self.selected_file = None  # Track selected file for rename/delete
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

        # Horizontal layout for combining the label_layout and buttons
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

        # Set up the context menu policy
        self.table_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.open_context_menu)

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

    def open_context_menu(self, position):
        """Show custom context menu on right-click"""
        index = self.table_widget.indexAt(position)
        if not index.isValid():
            return

        # Get the selected file or folder
        self.selected_file = self.table_widget.item(index.row(), 0).text()

        # Create a context menu
        context_menu = QtWidgets.QMenu(self)

        # Add Rename option
        rename_action = context_menu.addAction("Rename")
        rename_action.triggered.connect(self.rename_file)

        # Add Delete option
        delete_action = context_menu.addAction("Delete")
        delete_action.triggered.connect(self.delete_file)

        # Show the context menu at the clicked position
        context_menu.exec_(self.table_widget.viewport().mapToGlobal(position))

    def rename_file(self):
        """Rename the selected file"""
        if self.selected_file:
            # Get the full path of the selected file
            old_path = os.path.join(self.current_directory, self.selected_file)

            # Open a custom-styled input dialog to get the new name
            new_name, ok = self.create_custom_input_dialog("Rename File/Folder", "Enter new name:", self.selected_file)
            if ok and new_name:
                new_path = os.path.join(self.current_directory, new_name)
                try:
                    os.rename(old_path, new_path)
                    self.load_files(self.current_directory)  # Reload the files
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Rename Failed", f"Failed to rename file: {str(e)}")

    def delete_file(self):
        """Delete the selected file"""
        if self.selected_file:
            # Get the full path of the selected file
            file_path = os.path.join(self.current_directory, self.selected_file)

            # Ask for confirmation before deleting
            reply = self.create_custom_message_box("Delete File/Folder",
                                                   f"Are you sure you want to delete '{self.selected_file}'?",
                                                   "Delete", "Cancel")

            if reply == QtWidgets.QMessageBox.Yes:
                try:
                    if os.path.isdir(file_path):
                        os.rmdir(file_path)  # For directories, make sure they're empty
                    else:
                        os.remove(file_path)  # Delete file
                    self.load_files(self.current_directory)  # Reload the files
                except Exception as e:
                    self.create_custom_message_box("Delete Failed", f"Failed to delete file: {str(e)}", "Ok")

    def create_custom_message_box(self, title, message, yes_button_text="Yes", no_button_text="No"):
        """Create a QMessageBox with custom-styled buttons"""
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        # Customize QMessageBox buttons
        yes_button = msg_box.addButton(yes_button_text, QtWidgets.QMessageBox.YesRole)
        no_button = msg_box.addButton(no_button_text, QtWidgets.QMessageBox.NoRole)

        # Apply custom styles to buttons and message box
        msg_box.setStyleSheet("""
            QMessageBox {
                font-size: 12px;
                font-weight: bold;
                padding: 20px;
                font-family: 'Montserrat', sans-serif;
            }
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
        msg_box.exec_()

        return msg_box.clickedButton()

    def create_custom_input_dialog(self, title, label, text=""):
        """Create a QInputDialog with custom styling"""
        input_dialog = QtWidgets.QInputDialog(self)
        input_dialog.setWindowTitle(title)
        input_dialog.setLabelText(label)
        input_dialog.setTextValue(text)

        # Apply custom styles to input dialog
        input_dialog.setStyleSheet("""
            QMessageBox {
                font-size: 11px;
                font-family: 'Montserrat', sans-serif;
            }
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
            QLineEdit {
                padding: 5px;
                width: 500px;
                font-family: 'Montserrat', sans-serif;
                font-size: 11px;
            }
            QLabel {
                font-size: 12px; 
                font-weight: bold;
                font-family: 'Montserrat', sans-serif;
            }
        """)
        ok = input_dialog.exec_()

        return input_dialog.textValue(), ok

    def change_directory(self):
        """Change the currently selected directory"""
        new_directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", self.current_directory)
        if new_directory:
            self.current_directory = new_directory
            self.path_label.setText(self.current_directory)
            self.load_files(self.current_directory)  # Reload files from the new directory


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Local()
    window.show()
    sys.exit(app.exec_())
