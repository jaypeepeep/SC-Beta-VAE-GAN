from PyQt5 import QtCore, QtGui, QtWidgets
from components.widget.icon_only_widget import IconOnlyWidget  # Import the separated class
from components.widget.full_menu_widget import FullMenuWidget
from pages.workplace import Workplace
from pages.handwriting import Handwriting
from pages.about import About
from pages.local import Local

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1420, 820)
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

        # Menu Widget with Centered Title and Subtitle
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_5.setContentsMargins(0, 3, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        # Horizontal Layout for Menu Button, Title, and Subtitle
        self.header_widget = QtWidgets.QWidget(self.widget_3)
        self.header_layout = QtWidgets.QHBoxLayout(self.header_widget)
        # self.header_layout.setContentsMargins(5, 2, 5, 2)
        self.header_layout.setSpacing(0)
        self.header_layout.setObjectName("header_layout")

        # Menu Button
        self.change_btn = QtWidgets.QPushButton(self.header_widget)
        self.change_btn.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("./icon/menu-4-32.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.change_btn.setIcon(icon6)
        self.change_btn.setIconSize(QtCore.QSize(14, 14))
        self.change_btn.setCheckable(True)
        self.change_btn.setObjectName("change_btn")
        self.header_layout.addWidget(self.change_btn)
        self.change_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Spacer Before Title (Left Spacer)
        self.left_spacer = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.header_layout.addItem(self.left_spacer)

        # Vertical Layout for Title and Subtitle
        self.title_subtitle_layout = QtWidgets.QVBoxLayout()
        self.title_subtitle_layout.setSpacing(0)
        self.title_subtitle_layout.setAlignment(QtCore.Qt.AlignCenter)

        # "SC-β-VAE-GAN" Title
        self.title_label_1 = QtWidgets.QLabel(self.header_widget)
        self.title_label_1.setText("SC-β-VAE-GAN")
        self.title_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label_1.setStyleSheet("font-weight: bold; color: #003333; font-size: 14px;")
        self.title_subtitle_layout.addWidget(self.title_label_1)

        # Subtitle "Synthetic Data Generator"
        self.subtitle_label = QtWidgets.QLabel(self.header_widget)
        self.subtitle_label.setText("Synthetic Data Generator")
        self.subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("font-weight: bold; color: #003333; font-size: 15px;")
        self.title_subtitle_layout.addWidget(self.subtitle_label)

        # Add Title and Subtitle Layout to Header Layout
        self.header_layout.addLayout(self.title_subtitle_layout)

        # Spacer After Title (Right Spacer)
        self.right_spacer = QtWidgets.QSpacerItem(100, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.header_layout.addItem(self.right_spacer)

        # Add Header Layout to Vertical Layout
        self.verticalLayout_5.addWidget(self.header_widget)

        # Menu Button and Spacer
        self.widget = QtWidgets.QWidget(self.widget_3)
        self.widget.setMinimumSize(QtCore.QSize(0, 8))
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        # Stack Widgets
        self.verticalLayout_5.addWidget(self.widget)
        self.stackedWidget = QtWidgets.QStackedWidget(self.widget_3)
        self.stackedWidget.setObjectName("stackedWidget")

        # Page 1
        self.page1 = Workplace()
        self.stackedWidget.addWidget(self.page1)

        # Page 2
        self.page2 = Handwriting()
        self.stackedWidget.addWidget(self.page2)

        # Page 3
        self.page3 = About()
        self.stackedWidget.addWidget(self.page3)

        # Page 4
        self.page4 = Local()
        self.stackedWidget.addWidget(self.page4)
        
        self.verticalLayout_5.addWidget(self.stackedWidget)
        self.gridLayout.addWidget(self.widget_3, 0, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(3)
        self.change_btn.toggled['bool'].connect(self.icon_only_widget.setVisible)
        self.change_btn.toggled['bool'].connect(self.full_menu_widget.setHidden)
        self.icon_only_widget.workplace_btn_1.toggled['bool'].connect(self.full_menu_widget.workplace_btn_2.setChecked)
        self.icon_only_widget.handwriting_btn_1.toggled['bool'].connect(self.full_menu_widget.handwriting_btn_2.setChecked)
        self.icon_only_widget.about_btn_1.toggled['bool'].connect(self.full_menu_widget.about_btn_2.setChecked)
        self.icon_only_widget.local_btn_1.toggled['bool'].connect(self.full_menu_widget.local_btn_2.setChecked)
        self.full_menu_widget.workplace_btn_2.toggled['bool'].connect(self.icon_only_widget.workplace_btn_1.setChecked)
        self.full_menu_widget.handwriting_btn_2.toggled['bool'].connect(self.icon_only_widget.handwriting_btn_1.setChecked) 
        self.full_menu_widget.about_btn_2.toggled['bool'].connect(self.icon_only_widget.about_btn_1.setChecked) 
        self.full_menu_widget.local_btn_2.toggled['bool'].connect(self.icon_only_widget.local_btn_1.setChecked)
        self.full_menu_widget.exit_btn_2.clicked.connect(MainWindow.close)
        self.icon_only_widget.exit_btn_1.clicked.connect(MainWindow.close) 
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.full_menu_widget.workplace_btn_2.setText(_translate("MainWindow", "Workplace"))
        self.full_menu_widget.handwriting_btn_2.setText(_translate("MainWindow", "Handwriting"))
        self.full_menu_widget.about_btn_2.setText(_translate("MainWindow", "About"))
        self.full_menu_widget.local_btn_2.setText(_translate("MainWindow", "Local Storage"))
        self.full_menu_widget.exit_btn_2.setText(_translate("MainWindow", "Exit"))
import resource_rc
