from PyQt5 import QtCore, QtGui, QtWidgets
from components.icon_only_widget import IconOnlyWidget
class FullMenuWidget(QtWidgets.QWidget):
    def __init__(self,parent):
        super(FullMenuWidget, self).__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        self.setObjectName("text_icon_widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        
        #LOGO LABEL FOR TEXT AND ICON
        self.logo_label_2 = QtWidgets.QLabel(self)
        self.logo_label_2.setMinimumSize(QtCore.QSize(100, 100))
        self.logo_label_2.setMaximumSize(QtCore.QSize(100, 100))
        self.logo_label_2.setText("")
        self.logo_label_2.setPixmap(QtGui.QPixmap("./icon/Logo.png"))
        self.logo_label_2.setScaledContents(True)
        self.logo_label_2.setObjectName("logo_label_2")
        self.horizontalLayout_2.addWidget(self.logo_label_2)
        self.logo_label_3 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.logo_label_3.setFont(font)
        self.logo_label_3.setObjectName("logo_label_3")
        self.horizontalLayout_2.addWidget(self.logo_label_3)
        
        #Vertical layout
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        #Workplace Button
        self.workplace_btn_2 = QtWidgets.QPushButton(self)
        workIcon = QtGui.QIcon()
        workIcon.addPixmap(QtGui.QPixmap("./icon/home inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        workIcon.addPixmap(QtGui.QPixmap("./icon/home active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.workplace_btn_2.setIcon(workIcon)
        self.workplace_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.workplace_btn_2.setCheckable(True)
        self.workplace_btn_2.setAutoExclusive(True)
        self.workplace_btn_2.setObjectName("workplace_btn_2")
        self.verticalLayout_2.addWidget(self.workplace_btn_2)
        
        #Handwriting Button
        self.handwriting_btn_2 = QtWidgets.QPushButton(self)
        handIcon = QtGui.QIcon()
        handIcon.addPixmap(QtGui.QPixmap("./icon/hand inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        handIcon.addPixmap(QtGui.QPixmap("./icon/hand active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.handwriting_btn_2.setIcon(handIcon)
        self.handwriting_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.handwriting_btn_2.setCheckable(True)
        self.handwriting_btn_2.setAutoExclusive(True)
        self.handwriting_btn_2.setObjectName("handwriting_btn_2")
        self.verticalLayout_2.addWidget(self.handwriting_btn_2)
        
        #About Button
        self.about_btn_2 = QtWidgets.QPushButton(self)
        aboutIcon = QtGui.QIcon()
        aboutIcon.addPixmap(QtGui.QPixmap("./icon/about inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        aboutIcon.addPixmap(QtGui.QPixmap("./icon/about active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.about_btn_2.setIcon(aboutIcon)
        self.about_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.about_btn_2.setCheckable(True)
        self.about_btn_2.setAutoExclusive(True)
        self.about_btn_2.setObjectName("about_btn_2")
        self.verticalLayout_2.addWidget(self.about_btn_2)
        
        self.local_btn_2 = QtWidgets.QPushButton(self)
        localIcon = QtGui.QIcon()
        localIcon.addPixmap(QtGui.QPixmap("./icon/local inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        localIcon.addPixmap(QtGui.QPixmap("./icon/local active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.local_btn_2.setIcon(localIcon)
        self.local_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.local_btn_2.setCheckable(True)
        self.local_btn_2.setAutoExclusive(True)
        self.local_btn_2.setObjectName("local_btn_2")
        self.verticalLayout_2.addWidget(self.local_btn_2)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        
        #spacer Item
        spacerItem1 = QtWidgets.QSpacerItem(20, 450, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        
        #Exit Button
        self.exit_btn_2 = QtWidgets.QPushButton(self)
        exitIcon = QtGui.QIcon()
        exitIcon.addPixmap(QtGui.QPixmap("./icon/close.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.exit_btn_2.setIcon(exitIcon)
        self.exit_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.exit_btn_2.setObjectName("exit_btn_2")
        self.verticalLayout_4.addWidget(self.exit_btn_2)