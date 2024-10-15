from PyQt5 import QtWidgets, QtGui, QtCore
import sys


class SpinBoxWidget(QtWidgets.QWidget):
    def __init__(self, min_value, parent=None):
        super(SpinBoxWidget, self).__init__(parent)
        self.min_value = min_value
        self.setupUi()

    def setupUi(self):        
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)  # Add some margins to the layout

        self.label = QtWidgets.QLabel("Augmented Data Quantity")
        self.label.setStyleSheet(
            "font-family: Montserrat; font-size: 16px; font-weight: bold; color: #003333;"
        )
        self.layout.addWidget(self.label, alignment=QtCore.Qt.AlignLeft)

        # Create the number input (QSpinBox)
        self.number_input = QtWidgets.QSpinBox(self)
        self.number_input.setMinimum(self.min_value)
        self.number_input.setValue(self.min_value)
        self.number_input.setSingleStep(1)  # Set the step value to 1

        # Apply custom styles to the number input
        self.number_input.setStyleSheet(
            """
            QSpinBox {
                border-radius: 8px;
                border: 1px solid #003333;
                background: #f7f7f7;
                width: 80px;
                height: 30px;
                color: #000;
                font-family: Montserrat;
                font-size: 14px;
                font-weight: 600;
                text-align: center;
                padding: 5px;
            }
            QSpinBox::up-button {
                width: 25px;
                height: 25px;
                border: none;
                background-color: #f0f0f0;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QSpinBox::down-button {
                width: 25px;
                height: 20px;
                border: none;
                background-color: #f0f0f0;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QSpinBox::up-button:hover,
            QSpinBox::down-button:hover {
                background-color: #d0d0d0;
            }
            QSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed {
                background-color: #c0c0c0;
            }
            QSpinBox::up-arrow {
                image: url(icon/arrow_up.png);
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-arrow {
                image: url(icon/arrow_down.png);
                width: 10px;
                height: 10px;
            }
            """
        )

        self.layout.addWidget(self.number_input)

    def getValue(self):
        return self.number_input.value()
    
    def resetValue(self):
        self.number_input.setValue(self.min_value)