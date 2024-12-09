from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout


class FloatInputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Entropy Threshold")

        # 创建输入提示标签和输入框
        label = QLabel("information entropy threshold:")
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("for example: 0.5")

        # 设置输入验证器，确保输入为0到1之间的浮点数
        validator = QDoubleValidator(0.0, 1.0, 2)
        self.input_field.setValidator(validator)

        # 创建按钮
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        # 连接按钮事件
        ok_button.clicked.connect(self.accept)
        ok_button.setFixedSize(150, 35)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setFixedSize(150, 35)

        # 布局设置
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.input_field)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        # 将按钮布局添加到主布局
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_value(self):
        return self.input_field.text()


