from PyQt5 import QtCore, QtGui, QtWidgets
from components.icon_only_widget import IconOnlyWidget  # Import the separated class
from components.full_menu_widget import FullMenuWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        
        # Add the IconOnlyWidget
        self.icon_only_widget = IconOnlyWidget(self.centralwidget)
        self.gridLayout.addWidget(self.icon_only_widget, 0, 0, 1, 1)
        
        # Add the FullMenuWidget
        self.full_menu_widget = FullMenuWidget(self.centralwidget)
        self.gridLayout.addWidget(self.full_menu_widget, 0, 1, 1, 1)
        
        MainWindow.setCentralWidget(self.centralwidget)

    def retranslateUi(self, MainWindow):
        pass
