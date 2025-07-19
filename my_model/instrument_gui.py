from PyQt5.QtWidgets import QMainWindow, QApplication, QComboBox, QLabel, QFrame, QGridLayout, QGraphicsGridLayout, QScrollArea, QWidget, QPushButton
from PyQt5 import uic
from PyQt5.QtCore import QRect, QSize, Qt, QCoreApplication
from PyQt5.QtGui import QFont
import sys
import os

class CustomKeys(QMainWindow):
    def __init__(self, labels):
        super(CustomKeys, self).__init__()

        labels.sort() # Sort the labels
        self.labels = labels

        # Load the ui file
        uic.loadUi("gui_design.ui", self)

        # Set window title
        self.setWindowTitle("Customise sounds")

        # Create ScrollArea
        self.scrollArea = QScrollArea(self.centralwidget)
        # self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(40, 80, 421, 371))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 419, 369))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout = QGridLayout(self.scrollAreaWidgetContents)

        # Set title label
        self.title_label = self.findChild(QLabel, "label_3") 
        self.title_label.setFont(QFont('Calibri', 18)) # Set font

        # Create submit button
        self.submit_button = self.findChild(QPushButton, "pushButton")
        self.submit_button.clicked.connect(self.returnData)

        # Show on screen
        self.show()

        # Add a box for every label detected
        for i in range(len(self.labels)):
            self.createNewAssignment(i)

    def createNewAssignment(self, rowNum):
        # Create a frame for every label
        frame_name = "frame" + str(rowNum) # Give each frame a unique name
        self.frame = QFrame(self.scrollAreaWidgetContents)
        self.frame.setObjectName(frame_name) # to give frame its unique identifier
        # self.frame.setGeometry(QRect(50, 90, 400, 70))
        self.frame.setMinimumSize(QSize(400, 80))
        self.frame.setMaximumSize(QSize(400, 80))
        self.frame.setStyleSheet(u"background-color: rgb(174, 198, 207);")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)

        # self.gridLayout.setObjectName(u"gridLayout")
        # self.label = QLabel(self.frame)
        # self.label.setObjectName(u"label")

        self.subtitle_label = QLabel(self.frame)
        self.subtitle_label.setObjectName(u"label")
        self.subtitle_label.setGeometry(QRect(20, 5, 150, 20))
        self.subtitle_label.setFont(QFont('Calibri', 12))

        self.instrument_label = QLabel(self.frame)
        self.instrument_label.setObjectName(u"label")
        self.instrument_label.setGeometry(QRect(20, 30, 150, 20))
        self.instrument_label.setFont(QFont('Calibri', 8))

        self.note_label = QLabel(self.frame)
        self.note_label.setObjectName(u"label_2")
        self.note_label.setGeometry(QRect(220, 30, 160, 20))
        self.note_label.setFont(QFont('Calibri', 8))

        self.instrument_picker = QComboBox(self.frame)
        self.instrument_picker.setObjectName(u"comboBox")
        self.instrument_picker.setGeometry(QRect(20, 50, 160, 20))

        self.note_picker = QComboBox(self.frame)
        self.note_picker.setObjectName(u"comboBox_2")
        self.note_picker.setGeometry(QRect(220, 50, 140, 20))
        
        self.instrument_label.setText(QCoreApplication.translate("MainWindow", u"Choose an instrument:", None))
        self.note_label.setText(QCoreApplication.translate("MainWindow", u"Choose a note:", None))
        self.subtitle_label.setText(QCoreApplication.translate("MainWindow", u"Label: " + self.labels[rowNum], None))

        # Add instruments
        folder_path = os.path.join("notes")
        folders = os.listdir(folder_path)
        for instrument in folders:
            sound_path = os.path.join("notes", instrument)
            files = os.listdir(sound_path)
            file_names = []
            for f in files:
                if os.path.isfile(os.path.join(sound_path, f)):
                    file_names.append(f[:-4])
            self.instrument_picker.addItem(instrument, file_names)
        
        # Set defaults
        self.note_picker.setObjectName("note"+str(rowNum))
        self.instrument_picker.setObjectName(str(rowNum))
        self.instrument_picker.setCurrentIndex(0)
        self.note_picker.addItems(self.instrument_picker.itemData(0))

        # Activate dependent combobox
        self.instrument_picker.activated.connect(self.selected)

        self.gridLayout.addWidget(self.frame, rowNum, 0, 1, 1, Qt.AlignHCenter|Qt.AlignVCenter)

    def selected(self, index):
        # Identify which instrument_picker combobox changed
        sender = self.sender()
        id = "note" + sender.objectName()
        
        # Identify second combobox based on unique object name
        note_picker = self.scrollAreaWidgetContents.findChild(QComboBox, id)
        if note_picker:
            note_picker.clear() # Clear the second box
            # Update combobox accordingly
            note_picker.addItems(self.instrument_picker.itemData(index))

    def returnData(self):
        data = [] # List to store data
        
        for j in range(len(self.labels)):
            values = {} # Store subdata in a dictionary
            # Store label
            values['label'] = self.labels[j]
            # Store instrument chosen
            instrument = self.scrollAreaWidgetContents.findChild(QComboBox, str(j))
            values['instrument'] = instrument.currentText()
            # Store note chosen
            id = "note" + str(j)
            note = self.scrollAreaWidgetContents.findChild(QComboBox, id)
            values['note'] = note.currentText()
            data.append(values)

        # print(data)
        self.data = data # Assigns data to be accessed by object attribute - no need to return
        self.close() # Closes UI window
        # return data

if __name__ == "__main__":
    labels = ['5', '4', '1', '6', '0', '7', '8', '3', '2', '9']

    app = QApplication(sys.argv)
    UIWindow = CustomKeys(labels)

    app.exec_()
    print("retrieved data: ", UIWindow.data)
