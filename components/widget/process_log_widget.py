from PyQt5 import QtWidgets, QtGui, QtCore
import logging
import queue

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
                if self.widget:  # Ensure widget exists
                    self.widget.append_log(msg + '\n')
            except queue.Empty:
                break

class ProcessLogWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ProcessLogWidget, self).__init__(parent)
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
            QTextEdit QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background: #003333; 
                min-height: 30px;
                border-radius: 4px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background: #005555;
            }
            QTextEdit QScrollBar::add-line:vertical, 
            QTextEdit QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
            }
            QTextEdit QScrollBar::add-page:vertical, 
            QTextEdit QScrollBar::sub-page:vertical {
                background: transparent;
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

    def append_log(self, msg):
        if not self.text_preview_log:  # Check if widget still exists
            return
        try:
            if self.text_preview_log is not None:  # Add this check to ensure the widget exists
                self.text_preview_log.append(msg)
                self.text_preview_log.moveCursor(QtGui.QTextCursor.End)
        except RuntimeError as e:
            print(f"Error appending log: {e}")


    def clear(self):
        self.text_preview_log.clear()

    def get_logger(self):
        """Return the actual logger, allowing use of .info(), .error(), etc."""
        return self.logger