from PyQt5 import QtCore, QtGui, QtWidgets

class FullMenuWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(FullMenuWidget, self).__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        self.setObjectName("full_menu_widget")
        # Vertical layout for the whole widget
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self)# For the whole vertical layout
        self.verticalLayout_4.setContentsMargins(15, 15, 15, 15)
        self.setStyleSheet("""
            #full_menu_widget {
                background-color: #003333;
            }
            QPushButton {
                padding-left: 10px; 
                padding-right: 10px; 
            }
        """)

        # LOGO LABEL FOR TEXT AND ICON
        # Icon logo label
        self.logo_label_2 = QtWidgets.QLabel(self)
        self.logo_label_2.setMinimumSize(QtCore.QSize(100, 100))
        self.logo_label_2.setMaximumSize(QtCore.QSize(100, 100))
        self.logo_label_2.setText("")
        self.logo_label_2.setPixmap(QtGui.QPixmap("./icon/Logo.png"))
        self.logo_label_2.setScaledContents(True)
        self.logo_label_2.setObjectName("logo_label_2")
        self.verticalLayout_4.addWidget(self.logo_label_2, alignment=QtCore.Qt.AlignCenter)
        
        # Text logo label
        self.logo_label_3 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.logo_label_3.setFont(font)
        self.logo_label_3.setObjectName("logo_label_3")
        self.logo_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout_4.addWidget(self.logo_label_3, alignment=QtCore.Qt.AlignCenter)
        
        # Workplace Button
        self.workplace_btn_2 = QtWidgets.QPushButton(self)
        workIcon = QtGui.QIcon()
        workIcon.addPixmap(QtGui.QPixmap("./icon/home inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        workIcon.addPixmap(QtGui.QPixmap("./icon/home active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.workplace_btn_2.setIcon(workIcon)
        self.workplace_btn_2.setText("Hello")
        self.workplace_btn_2.setIconSize(QtCore.QSize(20, 20))
        self.workplace_btn_2.setCheckable(True)
        self.workplace_btn_2.setAutoExclusive(True)
        self.workplace_btn_2.setObjectName("workplace_btn_2")
        self.verticalLayout_4.addWidget(self.workplace_btn_2)
        # Handwriting Button
        self.handwriting_btn_2 = QtWidgets.QPushButton(self)
        handIcon = QtGui.QIcon()
        handIcon.addPixmap(QtGui.QPixmap("./icon/hand inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        handIcon.addPixmap(QtGui.QPixmap("./icon/hand active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.handwriting_btn_2.setIcon(handIcon)
        self.handwriting_btn_2.setIconSize(QtCore.QSize(20, 20))
        self.handwriting_btn_2.setCheckable(True)
        self.handwriting_btn_2.setAutoExclusive(True)
        self.handwriting_btn_2.setObjectName("handwriting_btn_2")
        self.verticalLayout_4.addWidget(self.handwriting_btn_2)
        
        # About Button
        self.about_btn_2 = QtWidgets.QPushButton(self)
        aboutIcon = QtGui.QIcon()
        aboutIcon.addPixmap(QtGui.QPixmap("./icon/about inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        aboutIcon.addPixmap(QtGui.QPixmap("./icon/about active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.about_btn_2.setIcon(aboutIcon)
        self.about_btn_2.setIconSize(QtCore.QSize(20, 20))
        self.about_btn_2.setCheckable(True)
        self.about_btn_2.setAutoExclusive(True)
        self.about_btn_2.setObjectName("about_btn_2")
        self.verticalLayout_4.addWidget(self.about_btn_2)
        
        # Local Button
        self.local_btn_2 = QtWidgets.QPushButton(self)
        localIcon = QtGui.QIcon()
        localIcon.addPixmap(QtGui.QPixmap("./icon/local inactive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        localIcon.addPixmap(QtGui.QPixmap("./icon/local active.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.local_btn_2.setIcon(localIcon)
        self.local_btn_2.setIconSize(QtCore.QSize(20, 20))
        self.local_btn_2.setCheckable(True)
        self.local_btn_2.setAutoExclusive(True)
        self.local_btn_2.setObjectName("local_btn_2")
        self.verticalLayout_4.addWidget(self.local_btn_2)
        
        # Spacer Item
        spacerItem1 = QtWidgets.QSpacerItem(20, 450, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        
        # Exit Button
        self.exit_btn_2 = QtWidgets.QPushButton(self)
        exitIcon = QtGui.QIcon()
        exitIcon.addPixmap(QtGui.QPixmap("./icon/close.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.exit_btn_2.setIcon(exitIcon)
        self.exit_btn_2.setIconSize(QtCore.QSize(20, 20))
        self.exit_btn_2.setObjectName("exit_btn_2")
        self.verticalLayout_4.addWidget(self.exit_btn_2)
        
        font = QtGui.QFont()
        font.setPointSize(8)  
        font.setBold(True)     
        self.workplace_btn_2.setFont(font)
        self.handwriting_btn_2.setFont(font)
        self.about_btn_2.setFont(font)
        self.local_btn_2.setFont(font)
        self.exit_btn_2.setFont(font)
        
        #set cursor to pointer
        self.workplace_btn_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.handwriting_btn_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.about_btn_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.local_btn_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exit_btn_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
    def paintEvent(self, event):
        # This ensures that the widget's background is painted
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)
        
        

