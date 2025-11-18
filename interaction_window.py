from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QSpacerItem, QSizePolicy
)
import sys


letter_to_feature = {'A': 'Supermarket', 'B': 'Milk Tea Shop', 'C': 'Pharmacy', 'D': 'Restaurant',
                     'E': 'Clothing Store', 'F': 'Antique Shop', 'G': 'Barber Shop', 'H': 'Electronic Store',
                     'I': 'Gift Shop', 'J': 'Theme Park', 'K': 'Fruit Shop', 'L': 'Cafe', 'M': 'Optical Shop',
                     'N': 'Dessert Shop', 'O': 'Movie Theater', 'P': 'Beauty Salon', 'Q': 'Bookstore',
                     'R': 'Hotel', 'S': 'Flower Shop', 'T': 'Gym'}


class CoLocationPatternWidget(QWidget):
    # Custom signal to send the list of interest labels
    interestSubmitted = pyqtSignal(list)

    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns
        self.initUI()

        # List of labels
        self.interest_labels = None

    def initUI(self):
        # Set window title and size
        self.setWindowTitle("Interaction")
        self.resize(1280, 720)

        # Set the window icon
        self.setWindowIcon(QIcon("icons/interaction_icon.png"))

        # Main layout
        main_layout = QVBoxLayout()

        # Create header row
        header_layout = QHBoxLayout()
        label_title = QLabel("  co-location patterns")
        label_title.setStyleSheet("font-size: 20px")
        label_interest = QLabel("interest")
        label_interest.setStyleSheet("font-size: 20px")
        header_layout.addWidget(label_title)
        header_layout.addStretch()
        header_layout.addWidget(label_interest)
        main_layout.addLayout(header_layout)

        # Create samples layout
        samples_layout = QVBoxLayout()
        samples_widget = QWidget()
        samples_widget.setLayout(samples_layout)
        main_layout.addWidget(samples_widget)
        main_layout.setStretchFactor(samples_widget, 1)

        # Create a row for each pattern
        self.checkboxes = []
        for pattern in self.patterns:
            row_layout = QHBoxLayout()

            # Display pattern content
            label = QLabel()
            text = "{"
            for index in range(len(pattern)):
                if index == len(pattern) - 1:
                    text += letter_to_feature[pattern[index]] + "}"
                else:
                    text += letter_to_feature[pattern[index]] + ", "
            label.setText(text)
            label.setStyleSheet("font-size: 20px")
            row_layout.addWidget(label)
            row_layout.setStretchFactor(label, 1)

            # Add spacing
            row_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

            # Create checkbox
            checkbox = QCheckBox()
            row_layout.addWidget(checkbox)
            row_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))  # Add extra spacing

            # Add checkbox to the list for later access
            self.checkboxes.append(checkbox)
            samples_layout.addLayout(row_layout)

        # Create submit button
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_interest)
        submit_button.setFixedSize(150, 40)

        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        button_layout.addWidget(submit_button)
        button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(button_layout)

        # Set the layout
        self.setLayout(main_layout)

    def submit_interest(self):
        # Submit the user's selections
        self.interest_labels = [1 if checkbox.isChecked() else 0 for checkbox in self.checkboxes]
        # Emit custom signal, passing the list of interest labels as a parameter
        self.interestSubmitted.emit(self.interest_labels)
        # Close the window
        self.close()
