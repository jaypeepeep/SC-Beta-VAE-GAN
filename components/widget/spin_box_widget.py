from PyQt5 import QtWidgets, QtGui, QtCore
from font.dynamic_font_size import get_font_sizes
from PyQt5.QtGui import QFont

class SpinBoxWidget(QtWidgets.QWidget):
    def __init__(self, min_value, parent=None):
        super(SpinBoxWidget, self).__init__(parent)
        self.min_value = min_value
        self.setupUi()
        font_sizes = get_font_sizes()
        font_family = "Montserrat"
        content_font = QFont(font_family, font_sizes["content"])

    def setupUi(self):
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        self.label = QtWidgets.QLabel("Augmented Data Quantity")
        self.label.setStyleSheet(
            "font-family: Montserrat; font-weight: bold; color: #003333;"
        )
        self.layout.addWidget(self.label)

        self.number_input = QtWidgets.QSpinBox(self)
        self.number_input.setMinimum(self.min_value)
        self.number_input.setValue(self.min_value)
        self.number_input.setSingleStep(1)
        self.number_input.setFixedSize(80, 40)
        self.number_input.setStyleSheet(
            """
            QSpinBox {
                border: 1px solid #003333;
                background: #f7f7f7;
                color: #000;
                font-family: Montserrat;
                font-weight: 600;
                padding: 4px;
                border-radius: 0px;
            }
            QSpinBox::up-button {
                width: 20px;
                height: 20px;
                border: none;
                background-color: #f0f0f0;
                border-radius: 0px;
            }
            QSpinBox::down-button {
                width: 20px;
                height: 20px;
                border: none;
                background-color: #f0f0f0;
                border-radius: 0px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #d0d0d0;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background-color: #c0c0c0;
            }
            QSpinBox::up-arrow {
                image: url(icon/arrow_up.png);
                width: 8px;
                height: 8px;
            }
            QSpinBox::down-arrow {
                image: url(icon/arrow_down.png);
                width: 8px;
                height: 8px;
            }
            """
        )
        self.layout.addWidget(self.number_input)
        self.layout.addStretch()

    def getValue(self):
        return self.number_input.value()

    def resetValue(self):
        self.number_input.setValue(self.min_value)