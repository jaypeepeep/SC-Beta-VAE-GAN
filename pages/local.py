from PyQt5 import QtWidgets, QtGui, QtCore


class Local(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Local, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)

        # Main label
        self.label_4 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setText("Local Page")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    local_widget = Local()
    local_widget.show()
    sys.exit(app.exec_())
