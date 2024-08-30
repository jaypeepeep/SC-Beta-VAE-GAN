from PyQt5 import QtCore, QtGui, QtWidgets

class IconOnlyWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(IconOnlyWidget, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.setObjectName("icon_only_widget")
        
        self.setStyleSheet("""
            #icon_only_widget {
                background-color: #003333;
            }
        """)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.logo_label_1 = QtWidgets.QLabel(self)
        self.logo_label_1.setMinimumSize(QtCore.QSize(50, 50))
        self.logo_label_1.setMaximumSize(QtCore.QSize(50, 50))
        self.logo_label_1.setText("")
        self.logo_label_1.setPixmap(QtGui.QPixmap("./icon/Logo.png"))
        self.logo_label_1.setScaledContents(True)
        self.logo_label_1.setObjectName("logo_label_1")
        self.horizontalLayout_3.addWidget(self.logo_label_1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")

        # HOME BUTTON
        self.workplace_btn_1 = QtWidgets.QPushButton(self)
        self.workplace_btn_1.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./icon/home inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap("./icon/home active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.workplace_btn_1.setIcon(icon)
        self.workplace_btn_1.setIconSize(QtCore.QSize(20, 20))
        self.workplace_btn_1.setCheckable(True)
        self.workplace_btn_1.setAutoExclusive(True)
        self.workplace_btn_1.setObjectName("workplace_btn_1")
        self.verticalLayout.addWidget(self.workplace_btn_1)

        # Handwriting Button
        self.handwriting_btn_1 = QtWidgets.QPushButton(self)
        self.handwriting_btn_1.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("./icon/hand inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap("./icon/hand active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.handwriting_btn_1.setIcon(icon1)
        self.handwriting_btn_1.setIconSize(QtCore.QSize(20, 20))
        self.handwriting_btn_1.setCheckable(True)
        self.handwriting_btn_1.setAutoExclusive(True)
        self.handwriting_btn_1.setObjectName("handwriting_btn_1")
        self.verticalLayout.addWidget(self.handwriting_btn_1)

        # About Button
        self.about_btn_1 = QtWidgets.QPushButton(self)
        self.about_btn_1.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./icon/about inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap("./icon/about active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.about_btn_1.setIcon(icon2)
        self.about_btn_1.setIconSize(QtCore.QSize(20, 20))
        self.about_btn_1.setCheckable(True)
        self.about_btn_1.setAutoExclusive(True)
        self.about_btn_1.setObjectName("about_btn_1")
        self.verticalLayout.addWidget(self.about_btn_1)

        # Local Storage Button
        self.local_btn_1 = QtWidgets.QPushButton(self)
        self.local_btn_1.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("./icon/local inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3.addPixmap(QtGui.QPixmap("./icon/local active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.local_btn_1.setIcon(icon3)
        self.local_btn_1.setIconSize(QtCore.QSize(20, 20))
        self.local_btn_1.setCheckable(True)
        self.local_btn_1.setAutoExclusive(True)
        self.local_btn_1.setObjectName("local_btn_1")
        self.verticalLayout.addWidget(self.local_btn_1)

        self.verticalLayout_3.addLayout(self.verticalLayout)

        spacerItem = QtWidgets.QSpacerItem(30, 375, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)

        self.exit_btn_1 = QtWidgets.QPushButton(self)
        self.exit_btn_1.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("./icon/close.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.exit_btn_1.setIcon(icon5)
        self.exit_btn_1.setIconSize(QtCore.QSize(20, 20))
        self.exit_btn_1.setObjectName("exit_btn_1")
        self.verticalLayout_3.addWidget(self.exit_btn_1)
    
    def paintEvent(self, event):
        # This ensures that the widget's background is painted
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)
