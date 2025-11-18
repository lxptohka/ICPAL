from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout


class FloatInputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Entropy Threshold")

        # Create input label and input field
        label = QLabel("information entropy threshold:")
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("for example: 0.5")

        # Set input validator to ensure the input is a float between 0 and 1
        validator = QDoubleValidator(0.0, 1.0, 2)
        self.input_field.setValidator(validator)

        # Create buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        # Connect button events
        ok_button.clicked.connect(self.accept)
        ok_button.setFixedSize(150, 35)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setFixedSize(150, 35)

        # Set main layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.input_field)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        # Add button layout to main layout
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_value(self):
        return self.input_field.text()



