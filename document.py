from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QToolBar, QAction
import sys


class DocumentationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document")
        self.resize(1000, 800)  # 设置窗口大小

        # 设置窗口的icon
        self.setWindowIcon(QIcon("icons/document_window_icon.png"))

        # 创建一个 QTextEdit 用于显示文档内容
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)  # 设置为只读模式

        # 文档内容
        documentation_text = """
Software User Guide：

1. Open File: Click "Open File" in the toolbar to browse and select the CSV data file you want to process.
           
2. Set Parameters: Click "Set Information Entropy Threshold" in the toolbar to enter a floating-point number between 0 and 1 as the entropy threshold for the algorithm.
           
3. Start Algorithm: Click "Start Mining" in the toolbar to begin the algorithm. The system will first generate frequent patterns, then interact with you through an interactive window. If you are interested in a pattern, you can check the box next to it; if not, you can leave it unchecked. After the interaction, the patterns you are interested in will be displayed at the bottom of the system interface.

4. The display area at the bottom of the interface includes the following sections: the “Progress” section shows task status messages during system execution; the “Prevalent” section displays all frequent patterns after mining is complete; and the “Results” section shows patterns that may be of interest to you after each interaction.
        """

        # 将文档内容加载到 QTextEdit 中
        self.text_edit.setText(documentation_text)
        font = QFont()
        font.setPointSize(12)  # 设置字体大小为 10
        self.text_edit.setFont(font)

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)

        # 应用布局到窗口
        self.setLayout(layout)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("主窗口")
#         self.resize(600, 400)
#
#         # 创建工具栏
#         toolbar = QToolBar("Main Toolbar", self)
#         self.addToolBar(toolbar)
#
#         # 添加“文档”按钮到工具栏
#         documentation_action = QAction("打开文档", self)
#         documentation_action.triggered.connect(self.open_documentation_window)
#         toolbar.addAction(documentation_action)
#
#     def open_documentation_window(self):
#         # 打开文档窗口
#         self.documentation_window = DocumentationWindow()
#         self.documentation_window.show()
#
#
# # 主程序
# app = QApplication(sys.argv)
# window = MainWindow()
# window.show()
# sys.exit(app.exec_())
