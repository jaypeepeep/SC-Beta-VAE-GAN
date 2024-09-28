from PyQt5 import QtWidgets, QtGui, QtCore

class ProcessLogWidget(QtWidgets.QWidget):
    def __init__(self, filename=None, parent=None):
        super(ProcessLogWidget, self).__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)

        self.container_widget_log = QtWidgets.QWidget(self)
        self.container_layout_log = QtWidgets.QVBoxLayout(self.container_widget_log)
        self.container_layout_log.setContentsMargins(10, 10, 10, 10)
        self.container_widget_log.setStyleSheet(
            "background-color: #E0E0E0; border-radius: 5px; padding: 10px;"
        )
        self.layout.addWidget(self.container_widget_log)

        self.text_preview_log = QtWidgets.QTextEdit(self.container_widget_log)
        self.text_preview_log.setPlainText(
            "Epoch 1/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.91batch/s, loss=0.000568]\n"
            "Epoch 2/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000494]\n"
            "Epoch 3/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000807]\n"
            "Epoch 4/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000590]\n"
            "Epoch 5/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 6/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 7/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 8/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]\n"
            "Epoch 9/100: 100% ▉▉▉▉▉▉▉▉▉▉ 1329/1329 [00:20<00:00, 64.93batch/s, loss=0.000354]"
        )
        self.text_preview_log.setReadOnly(True)
        self.text_preview_log.setFixedHeight(300)
        self.text_preview_log.setStyleSheet(
            "background-color: white; border: 1px solid #dcdcdc; font-family: Montserrat; font-size: 12px;"
        )
        self.container_layout_log.addWidget(self.text_preview_log)