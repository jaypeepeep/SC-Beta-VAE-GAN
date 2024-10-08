from PyQt5 import QtWidgets, QtGui, QtCore
import logging
import queue
import time

class QTextEditLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.queue = queue.Queue()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_widget)
        self.timer.start(100)  # Update every 100ms

    def emit(self, record):
        self.queue.put(record)

    def update_widget(self):
        while not self.queue.empty():
            try:
                record = self.queue.get()
                msg = self.format(record)
                self.widget.append_log(msg + '\n')
            except queue.Empty:
                break

class ProcessLogWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_logger()

    def setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # Create container widget
        self.container_widget_log = QtWidgets.QWidget(self)
        self.container_layout_log = QtWidgets.QVBoxLayout(self.container_widget_log)
        self.container_layout_log.setContentsMargins(10, 10, 10, 10)
        self.container_widget_log.setStyleSheet(
            "background-color: #E0E0E0; border-radius: 5px; padding: 10px;"
        )
        
        # Create text preview widget
        self.text_preview_log = QtWidgets.QTextEdit(self.container_widget_log)
        self.text_preview_log.setReadOnly(True)
        self.text_preview_log.setFixedHeight(300)
        self.text_preview_log.setStyleSheet(
            """
            QTextEdit {
                background-color: white;
                border: 1px solid #dcdcdc;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 5px;
            }
            """
        )
        
        # Add text preview to container layout
        self.container_layout_log.addWidget(self.text_preview_log)
        
        # Add container to main layout
        self.layout.addWidget(self.container_widget_log)

    def setup_logger(self):
        # Create custom logger
        self.logger = logging.getLogger('ConsoleLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # Create handler and formatter
        self.log_handler = QTextEditLogger(self)
        self.log_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Add handler to logger
        self.logger.addHandler(self.log_handler)

    def append_log(self, text):
        self.text_preview_log.moveCursor(QtGui.QTextCursor.End)
        self.text_preview_log.insertPlainText(text)
        self.text_preview_log.moveCursor(QtGui.QTextCursor.End)

    def clear(self):
        self.text_preview_log.clear()

    def get_logger(self):
        return self.logger