from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QSpacerItem, QSizePolicy
)
import sys


# 捏特征
letter_to_feature = {'A': 'Supermarket', 'B': 'Milk Tea Shop', 'C': 'Pharmacy', 'D': 'Restaurant',
                     'E': 'Clothing Store', 'F': 'Antique Shop', 'G': 'Barber Shop', 'H': 'Electronic Store',
                     'I': 'Gift Shop', 'J': 'Theme Park', 'K': 'Fruit Shop', 'L': 'Cafe', 'M': 'Optical Shop',
                     'N': 'Dessert Shop', 'O': 'Movie Theater', 'P': 'Beauty Salon', 'Q': 'Bookstore',
                     'R': 'Hotel', 'S': 'Flower Shop', 'T': 'Gym'}


class CoLocationPatternWidget(QWidget):
    # 自定义信号，发送兴趣标签列表
    interestSubmitted = pyqtSignal(list)

    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns
        self.initUI()

        # 标签列表
        self.interest_labels = None

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle("Interaction")
        self.resize(1280, 720)

        # 设置窗口的icon
        self.setWindowIcon(QIcon("icons/interaction_icon.png"))

        # 主布局
        main_layout = QVBoxLayout()

        # 创建标题行
        header_layout = QHBoxLayout()
        label_title = QLabel("  co-location patterns")
        label_title.setStyleSheet("font-size: 20px")
        label_interest = QLabel("interest")
        label_interest.setStyleSheet("font-size: 20px")
        header_layout.addWidget(label_title)
        header_layout.addStretch()
        header_layout.addWidget(label_interest)
        main_layout.addLayout(header_layout)

        # 创建样本布局
        samples_layout = QVBoxLayout()
        samples_widget = QWidget()
        samples_widget.setLayout(samples_layout)
        main_layout.addWidget(samples_widget)
        main_layout.setStretchFactor(samples_widget, 1)

        # 为每个模式创建一行
        self.checkboxes = []
        for pattern in self.patterns:
            row_layout = QHBoxLayout()

            # 显示模式内容
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

            # 添加空格
            row_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

            # 创建复选框
            checkbox = QCheckBox()
            row_layout.addWidget(checkbox)
            row_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))  # 添加额外空格

            # 将复选框添加到列表以便后续访问
            self.checkboxes.append(checkbox)
            samples_layout.addLayout(row_layout)

        # 创建提交按钮
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_interest)
        submit_button.setFixedSize(150, 40)

        # 将按钮居中
        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        button_layout.addWidget(submit_button)
        button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(button_layout)

        # 设置布局
        self.setLayout(main_layout)

    def submit_interest(self):
        # 提交用户的选择
        self.interest_labels = [1 if checkbox.isChecked() else 0 for checkbox in self.checkboxes]
        # 发射自定义信号，将兴趣标签列表作为参数传递
        self.interestSubmitted.emit(self.interest_labels)
        # 关闭窗口
        self.close()


# # 主程序
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#
#     # 示例数据
#     patterns = [
#         ["F", "R"], ["E", "S"], ["B", "S", "T"], ["J", "O"],
#         ["A", "L", "M"], ["L", "O", "P"], ["K", "N"],
#         ["I", "N"], ["M", "R", "S"], ["B", "K", "L"],["B", "K", "L"],
#         ["B", "K", "L"], ["B", "K", "L"], ["B", "K", "L"], ["B", "K", "L"]
#     ]
#
#     window = CoLocationPatternWidget(patterns)
#     window.show()
#     sys.exit(app.exec_())
