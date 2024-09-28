from PyQt5 import QtWidgets, QtGui, QtCore
import sys


class SpinBoxWidget(QtWidgets.QWidget):
    def __init__(self, min_value, parent=None):
        super(SpinBoxWidget, self).__init__(parent)
        self.min_value = min_value
        self.setupUi()

    def setupUi(self):
        self.layout = QtWidgets.QHBoxLayout(self)

        self.label = QtWidgets.QLabel("Augmented Data Quantity")
        self.label.setStyleSheet(
            "font-family: Montserrat; font-size: 14px; font-weight: bold; text-align: center;"
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
                border-radius: 5px;
                border: 1px solid #000;
                background: #FFF;
                width: 50px;
                height: 20px;
                color: #000;
                font-family: Montserrat;
                font-size: 14px;
                font-weight: 600;
                text-align: center;
            }
            QSpinBox::up-button {
                width: 20px;
                height: 20px;
                margin-bottom: -1px; /* Adjust the margin */
                padding: 0;
            }
            QSpinBox::down-button {
                width: 20px;
                height: 20px;
                margin-top: -1px; /* Adjust the margin */
                padding: 0;
            }
            QSpinBox::up-button:hover,
            QSpinBox::down-button:hover {
                background-color: #e0e0e0;
            }
        """
        )

        self.layout.addWidget(self.number_input)
