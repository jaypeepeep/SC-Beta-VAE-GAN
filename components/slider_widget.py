# slider_widget.py

from PyQt5 import QtWidgets, QtGui, QtCore

class SliderWidget(QtWidgets.QWidget):
    def __init__(self, min_value=0, max_value=10, parent=None):
        super(SliderWidget, self).__init__(parent)
        self.min_value = min_value
        self.max_value = max_value
        self.setupUi()

    def setupUi(self):
        self.layout = QtWidgets.QHBoxLayout(self)

        # Create the slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(self.min_value)
        self.slider.setMaximum(self.max_value)
        self.slider.setValue(self.min_value)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.updateValue)

        # Apply custom styles to the slider
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border-radius: 5px;
                background: #CACACA;
                height: 10px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 2px solid black;
                width: 29px;
                height: 47px;
                border-radius: 4px;
                margin: -10px 0;
            }
        """)

        # Create the number input
        self.number_input = QtWidgets.QSpinBox(self)
        self.number_input.setMinimum(self.min_value)
        self.number_input.setMaximum(self.max_value)
        self.number_input.setValue(self.min_value)
        self.number_input.valueChanged.connect(self.updateSlider)

        # Apply custom styles to the number input
        self.number_input.setStyleSheet("""
            QSpinBox {
                border-radius: 5px;
                border: 3px solid #000;
                background: #FFF;
                width: 100px;
                height: 71px;
                color: #000;
                font-family: Montserrat;
                font-size: 20px;
                font-style: normal;
                font-weight: 600;
                line-height: normal;
                text-align: center;
                padding-left: 30px;
            }
            QSpinBox::up-button {
                width: 35px;
                height: 35px;
                margin-bottom: -1px; /* Adjust the margin to create space between buttons */
                padding: 0;
            }
            QSpinBox::down-button {
                width: 35px;
                height: 35px;
                margin-top: -1px; /* Adjust the margin to create space between buttons */
                padding: 0;
            }
            QSpinBox::up-button:hover,
            QSpinBox::down-button:hover {
                background-color: #e0e0e0;
            }
        """)

        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.number_input)

    def updateValue(self, value):
        self.number_input.setValue(value)

    def updateSlider(self, value):
        self.slider.setValue(value)
