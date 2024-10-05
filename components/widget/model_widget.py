import os
from components.widget.spin_box_widget import SpinBoxWidget
from PyQt5 import QtWidgets, QtGui, QtCore


class ModelWidget(QtWidgets.QWidget):
    def __init__(self, filename=None, parent=None):
        super(ModelWidget, self).__init__(parent)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.pre_trained_path = os.path.join(script_dir, "../../pre-trained")

        if not os.path.exists(self.pre_trained_path):
            os.makedirs(self.pre_trained_path)

        self.setup_ui()
        # Add a QTimer to refresh the file list every 5 seconds (5000 milliseconds)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.load_files)  # Connect the timer to the load_files method
        self.timer.start(5000)  # Refresh every 5 seconds

    def setup_ui(self):
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setContentsMargins(20, 10, 20, 20)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.setFont(font)

        self.setLayout(self.layout)

        # Buttons layout (Train VAE)
        self.train_button = QtWidgets.QPushButton("Train VAE")
        button_style = """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 14px; 
                font-weight: 600; 
                padding: 8px 16px;
                margin-left: 15px; 
                margin-right: 15px; 
                border-radius: 5px; 
                border: none;
            }
            QPushButton:hover {
                background-color: #005555;
            }
            QPushButton:disabled {
                background-color: #999999;  
                color: white;
            }
        """
        self.train_button.setStyleSheet(button_style)
        self.train_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Add the train button to the grid layout
        self.layout.addWidget(self.train_button)  # Row 0, Column 0
        
        # Table to display files in ../pre-trained
        self.files_table = QtWidgets.QTableWidget(self)
        self.files_table.setColumnCount(4)
        self.files_table.setHorizontalHeaderLabels(["Select", "Name", "Date", "Actions"])
        self.files_table.horizontalHeader().setStretchLastSection(True)  # Stretch the last column to fit


        # Set fixed height for the table
        self.files_table.setColumnWidth(1, 600)  # Adjusted "Name" column width
        self.files_table.setFixedHeight(600)  # Set the height to 600px
        
        font_x = QtGui.QFont()
        font_x.setPointSize(8)
        self.files_table.setFont(font_x)
        # Set uniform style for the table
        self.files_table.setStyleSheet("""    
            QHeaderView::section {
                background-color: transparent;
                padding: 5px;
                color: #033;            
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #033;
                min-height: 50px;           
            }

            QTableWidget::item {
                border: 1px solid #033;
                padding-left: 10px;
                padding-right: 10px;
            }

            QTableCornerButton::section {
                border: none;
                background-color: transparent;
            }
        """)

        # Add table to layout
        self.layout.addWidget(self.files_table)  # Row 1, Column 0

        # Slider widget in Input collapsible
        self.slider_widget = SpinBoxWidget(0)
        self.layout.addWidget(self.slider_widget)  # Row 2, Column 0

        # Fetch files from ../pre-trained and display
        self.load_files()

        # Connect button actions
        self.train_button.clicked.connect(self.train_vae)
    def load_files(self, directory=None):
        """Populate the table with files and folders from the given directory."""
        if directory is None:
            directory = self.pre_trained_path  # Use the pre-trained path if no directory is provided

        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Clear the table first
        self.files_table.setRowCount(0)

        # Get list of files and directories
        files = os.listdir(directory)
        self.files_table.setRowCount(len(files))  # Set the number of rows in the table

        for index, file_name in enumerate(files):
            # Get full path
            full_path = os.path.join(directory, file_name)

            # Set file/folder name
            name_item = QtWidgets.QTableWidgetItem(file_name)
            self.files_table.setItem(index, 1, name_item)  # Set to column 1 for Name

            # Get the modification date
            mod_time = os.path.getmtime(full_path)
            mod_date = QtCore.QDateTime.fromSecsSinceEpoch(int(mod_time)).toString(QtCore.Qt.DefaultLocaleShortDate)

            # Set modification date
            date_item = QtWidgets.QTableWidgetItem(mod_date)
            self.files_table.setItem(index, 2, date_item)  # Set to column 2 for Date

            # Add checkbox for selection
            checkbox = QtWidgets.QCheckBox(self.files_table)
            checkbox.setStyleSheet("""
                QCheckBox {
                    padding: auto;
                }
                QCheckBox::indicator {
                    width: 15px;
                    height: 15px;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #ccc;
                    border: 2px solid #003333;
                    border-radius: 3px;
                }
                QCheckBox::indicator:checked {
                    background-color: #005555;
                    border: 2px solid #003333;
                    border-radius: 3px;
                }
            """)
            self.files_table.setCellWidget(index, 0, checkbox)  # Column 0 for checkbox

            # Actions button with custom icon
            actions_button = QtWidgets.QPushButton()
            actions_button.setIcon(QtGui.QIcon('../../icon/arrow_up.png'))  # Add an icon for the button
            actions_button.setIconSize(QtCore.QSize(20, 20))  # Set the size of the icon
            actions_button.setFixedSize(40, 40)  # Set button size
            actions_button.setStyleSheet("""
                QPushButton {
                    border: none;
                    border-radius: 5px;
                    
                }
                QPushButton:hover {
                    background-color: yellow;
                }
            """)
            actions_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            actions_button.clicked.connect(lambda _, f=file_name: self.show_file_options(f))
            self.files_table.setCellWidget(index, 3, actions_button)  # Set to column 3 for Actions

        # Ensure all items are loaded properly
        self.files_table.viewport().update()

    def show_file_options(self, file):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu { 
                background-color: #033; 
                border: none;
                font-weight: bold;
            }
            QMenu::item { 
                color: white; 
            }
            QMenu::item:selected { 
                background-color: white;
                color: #033;
            }
            QMenu::item:hover { 
                background-color: white;
                color: #033; 
            }
        """)

        rename_action = QtWidgets.QAction("Rename", self)
        rename_action.triggered.connect(lambda: self.rename_file(file))

        delete_action = QtWidgets.QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.delete_file(file))

        menu.addAction(rename_action)
        menu.addAction(delete_action)

        menu.exec_(QtGui.QCursor.pos())

    def rename_file(self, file):
        """Rename a file using a custom input dialog"""
        new_name, ok = self.create_custom_input_dialog(
            title="Rename File",
            label=f"Enter new name for {file}:"
        )
        
        if ok and new_name:
            old_path = os.path.join(self.pre_trained_path, file)
            new_path = os.path.join(self.pre_trained_path, new_name)
            try:
                os.rename(old_path, new_path)
                self.refresh_file_list()
            except OSError as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to rename file: {str(e)}")

    def delete_file(self, file):
        """Delete a file after user confirms through a custom message box"""
        confirm_delete = self.create_custom_message_box(
            title="Delete File",
            message=f"Are you sure you want to delete {file}?"
        )
        
        if confirm_delete:
            file_path = os.path.join(self.pre_trained_path, file)
            try:
                os.remove(file_path)
                self.refresh_file_list()
            except OSError as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to delete file: {str(e)}")

    def train_vae(self):
        self.create_custom_message_box(
            title="Train VAE",
            message=f"Training VAE model..."
        )
    
    def create_custom_message_box(self, title, message):
        """Create a custom message box"""
        message_box = QtWidgets.QMessageBox()
        message_box.setWindowTitle(title)
        message_box.setText(message)
        message_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        message_box.setStyleSheet("""
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

        return message_box.exec_() == QtWidgets.QMessageBox.Yes
    

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
    
    def refresh_file_list(self):
        self.load_files()

