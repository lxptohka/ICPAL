import csv
import sys
import time

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSpacerItem, QSizePolicy,
    QLabel, QCheckBox, QPushButton, QTextEdit, QAction, QMainWindow, QLineEdit, QToolBar, QInputDialog, QMessageBox,
    QDialog, QStatusBar, QProgressBar, QTabWidget, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView
)

import random
from sklearn import svm
import math

from joinbased import get_prevalent_patterns
import data_distribution
import float_input_dialog
import document
from interaction_window import CoLocationPatternWidget


# 捏特征
letter_to_feature = {'A': 'Supermarket', 'B': 'Milk Tea Shop', 'C': 'Pharmacy', 'D': 'Restaurant',
                     'E': 'Clothing Store', 'F': 'Antique Shop', 'G': 'Barber Shop', 'H': 'Electronic Store',
                     'I': 'Gift Shop', 'J': 'Theme Park', 'K': 'Fruit Shop', 'L': 'Cafe', 'M': 'Optical Shop',
                     'N': 'Dessert Shop', 'O': 'Movie Theater', 'P': 'Beauty Salon', 'Q': 'Bookstore',
                     'R': 'Hotel', 'S': 'Flower Shop', 'T': 'Gym'}


class WorkerThread(QThread):
    task_completed = pyqtSignal(list)  # 定义信号，用于传递任务返回的结果

    def __init__(self, filename):
        super().__init__()
        self.filename = filename  # 接收参数并保存

    def run(self):
        result = get_prevalent_patterns(self.filename)  # 调用耗时任务，传入参数
        self.task_completed.emit(result)  # 任务完成后发出信号，传递返回值


class Demo(QMainWindow):
    """demo 主界面"""
    def __init__(self):
        super().__init__()
        # 数据文件路径
        self.filename = None
        # 频繁模式存储列表
        self.prevalent_patterns = None
        # 按模式阶数递增的频繁模式列表
        self.sorted_prevalent_patterns = None

        # 存储每次选择后的样本集
        self.samples = []
        # 存储每次选择后的样本集的标签
        self.results = []

        # 已标记样本集
        self.patterns = []  # 存储模式的列表
        self.labels = []  # 模式对应的兴趣标签

        # 特征数
        self.featureNum = 0
        # 特征在特征向量中对应的位置
        self.map = {}

        # 分类器
        self.classifier = svm.SVC(kernel='linear', C=1.0, random_state=1, probability=True)

        # 熵阈值，默认为0.9
        self.entropy_threshold = 0.9

        # 文档窗口
        self.documentation_window = None

        # 记录交互次数
        self.interact_times = 0

        self.initUI()

    def initUI(self):
        """设置界面"""
        # 设置窗口标题
        self.setWindowTitle('ICPAL')

        # 获取屏幕分辨率
        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()

        # 设置窗口大小为屏幕分辨率
        self.setGeometry(0, 0, screen_width, screen_height)

        # 设置窗口的icon
        self.setWindowIcon(QIcon("icons/icon.png"))

        # 创建工具栏
        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(toolbar)

        # 添加 open file 工具
        open_file_action = QAction(QIcon("icons/file_icon.png"), "Open File", self)
        open_file_action.triggered.connect(self.on_view_click)
        toolbar.addAction(open_file_action)

        # 添加一个自定义空白区域
        spacer1 = QWidget()
        spacer1.setFixedWidth(50)  # 设置空白区域的宽度
        toolbar.addWidget(spacer1)

        # 添加信息熵阈值设置工具
        threshold_action = QAction(QIcon("icons/threshold_icon.png"), "Set Information Entropy Threshold", self)
        threshold_action.triggered.connect(self.set_threshold)
        toolbar.addAction(threshold_action)

        # 添加一个自定义空白区域
        spacer2 = QWidget()
        spacer2.setFixedWidth(50)  # 设置空白区域的宽度
        toolbar.addWidget(spacer2)

        # 算法，启动！
        start_action = QAction(QIcon("icons/start_icon.png"), "Start Mining", self)
        start_action.triggered.connect(self.start_click)
        toolbar.addAction(start_action)

        # 添加一个自定义空白区域
        spacer3 = QWidget()
        spacer3.setFixedWidth(50)  # 设置空白区域的宽度
        toolbar.addWidget(spacer3)

        # 添加 Help 工具
        about_action = QAction(QIcon("icons/document_icon.png"), "Documentation", self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)

        # 创建中心小部件
        central_widget = QWidget()
        # 创建垂直布局
        self.layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # 创建用于与用户交互获取标签和显示数据分布的布局
        self.data_and_distribution_widget = QWidget()
        self.data_and_distribution_layout = QHBoxLayout()
        self.data_and_distribution_widget.setLayout(self.data_and_distribution_layout)

        # 创建用于显示原始数据的布局
        self.table_widget = QTableWidget()
        self.table_widget.setFixedSize(int(screen_width * 0.3), int(screen_height * 0.6))
        self.data_and_distribution_layout.addWidget(self.table_widget)

        # 设置表格列宽和行高为自适应模式
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 使列宽自适应表格宽度
        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 使行高自适应表格高度

        # 创建用于显示数据分布的布局
        self.data_widget = QWidget()
        self.data_widget.setFixedSize(int(screen_width * 0.6), int(screen_height * 0.6))
        self.data_layout = QHBoxLayout()
        self.data_widget.setLayout(self.data_layout)
        self.canvas = data_distribution.PlotCanvas(self, width=17, height=9)
        self.data_layout.addWidget(self.canvas)
        self.data_and_distribution_layout.addWidget(self.data_widget)
        self.layout.addWidget(self.data_and_distribution_widget)

        # 创建 QTabWidget 显示已执行的任务、频繁模式和用户可能感兴趣的模式
        self.tab_widget = QTabWidget()
        self.tab_widget.setFixedSize(screen_width, int(screen_height * 0.25))
        # 显示已执行的任务
        self.progressText = QTextEdit(self)
        self.progressText.setReadOnly(True)
        # 设置字体大小
        font = QFont()
        font.setPointSize(12)  # 设置字体大小为 12
        self.progressText.setFont(font)
        self.tab_widget.addTab(self.progressText, "Progress")
        # 显示频繁模式的文本框
        self.prevalentText = QTextEdit(self)
        self.prevalentText.setReadOnly(True)
        self.tab_widget.addTab(self.prevalentText, "Prevalent")
        # 显示用户感兴趣的模式的文本框
        self.patternText = QTextEdit(self)
        self.patternText.setReadOnly(True)
        self.tab_widget.addTab(self.patternText, "Results")
        self.layout.addWidget(self.tab_widget)

        # 设置状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 创建任务名称标签
        self.task_label = QLabel("Generating frequent patterns...")
        self.task_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.task_label.setVisible(False)  # 初始化时隐藏

        # 创建一个不确定进度的进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignRight)
        self.progress_bar.setFixedSize(200, 20)  # 设置进度条尺寸
        self.progress_bar.setVisible(False)  # 初始化时隐藏
        self.progress_bar.setRange(0, 0)  # 设置为不确定进度状态

        # 将任务标签和进度条添加到状态栏中
        self.status_bar.addPermanentWidget(self.task_label)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # 设置中心小部件和布局
        self.setCentralWidget(central_widget)
        self.setLayout(self.layout)

        # 设置窗口启动时最大化
        self.showMaximized()

    def show_about(self):
        """打开使用文档"""
        # 创建文档窗口
        if self.documentation_window is None:
            self.documentation_window = document.DocumentationWindow()
        # 显示文档窗口
        self.documentation_window.show()

    def set_threshold(self):
        """设置信息熵阈值"""
        # 弹出输入对话框，让用户输入浮点数
        while True:
            # 创建并显示自定义对话框
            dialog = float_input_dialog.FloatInputDialog()
            if dialog.exec_() == QDialog.Accepted:
                # 获取输入值并检查范围
                try:
                    value = float(dialog.get_value())
                    if 0 <= value <= 1:
                        QMessageBox.information(self, "Succeed",
                                                f"Successfully set the information entropy threshold to {value}.")
                        self.entropy_threshold = value
                        break
                    else:
                        QMessageBox.warning(self, "Error", "Please enter a floating-point number between 0 and 1!")
                except ValueError:
                    QMessageBox.warning(self, "Error", "Please enter a valid floating-point number!")
            else:
                break

    def on_view_click(self):
        """选择数据文件"""
        # 打开文件选择对话框
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "select CSV file", "", "CSV Files (*.csv);;All Files (*)",
                                                  options=options)
        # 获取文件路径
        self.filename = fileName
        # 显示执行信息
        self.progressText.setText("Successfully opened the file " + fileName)

        # 读取数据
        data = []
        with open(self.filename, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row[0]) > 1:
                    continue
                temp = [row[0], int(row[1]), float(row[2]), float(row[3])]
                data.append(temp)
        del data[0]

        # 读取并显示 CSV 文件内容
        with open(fileName, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            row = list(reader)  # 将 CSV 文件内容转换为列表

            # 设置表格行数和列数
            self.table_widget.setRowCount(len(row))
            self.table_widget.setColumnCount(len(row[0]))

            # 将数据插入表格
            for row_index, row_data in enumerate(row):
                if row_index != 0:
                    row_data[0] = letter_to_feature[row_data[0]]
                for col_index, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(cell_data)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # 设置表格数据为不可编辑
                    item.setTextAlignment(Qt.AlignCenter)  # 设置文本居中
                    self.table_widget.setItem(row_index, col_index, item)

        self.canvas.plot_points(data)

    def start_click(self):
        """开始执行程序"""
        # 显示任务名称和进度条
        self.task_label.setVisible(True)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()  # 强制更新界面

        # 添加执行信息
        self.progressText.append("Generating frequent patterns...")

        # 启动后台线程，并将参数传递给线程
        self.worker_thread = WorkerThread(self.filename)
        self.worker_thread.task_completed.connect(self.handle_task_result)  # 连接信号到结果处理函数
        # 获取开始生成频繁模式的时间
        self.start_time = time.time()
        self.worker_thread.start()

    def handle_task_result(self, result):
        """继续执行算法"""
        # 获取频繁模式生成完成的时间
        self.end_time = time.time()
        # 获取频繁模式
        self.prevalent_patterns = result
        self.sorted_prevalent_patterns = sorted(result, key=lambda x: len(x))

        # 添加执行信息
        self.progressText.append("Successfully generated frequent patterns, taking {:.2f} seconds."
                                 .format(self.end_time - self.start_time))

        # 显示频繁模式
        text = ""
        for prevalent_pattern in self.prevalent_patterns:
            text += "{"
            for index in range(len(prevalent_pattern)):
                if index == len(prevalent_pattern) - 1:
                    text += letter_to_feature[prevalent_pattern[index]] + "} "
                else:
                    text += letter_to_feature[prevalent_pattern[index]] + ", "
        # 设置字体大小
        font = QFont()
        font.setPointSize(12)  # 设置字体大小为 12
        self.prevalentText.setFont(font)
        self.prevalentText.setText(text)
        QApplication.processEvents()  # 强制更新界面

        # 任务完成时隐藏进度条和任务名称，并显示完成信息
        self.task_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Done.", 3000)  # 显示完成信息2秒钟

        # 获取频繁模式中出现的特征数和特征对应的特征向量位置
        self.get_featureNum_and_Map()
        # 将频繁模式随机打乱
        random.shuffle(self.prevalent_patterns)

        # 首次执行程序先随机采样一部分模式用作训练集
        k = 10
        self.samples = random.sample(self.prevalent_patterns, k)

        # 添加执行信息
        self.interact_times += 1
        self.progressText.append("Executing the 1st interaction...")

        # 弹出交互界面
        self.show_interaction_window()

    def show_interaction_window(self):
        """弹出交互界面"""
        self.interaction_window = CoLocationPatternWidget(self.samples)
        self.interaction_window.interestSubmitted.connect(self.get_interest_labels)
        self.interaction_window.show()

    def get_interest_labels(self, interest_labels):
        """获取交互结果"""
        self.results = interest_labels
        self.get_user_selection()

    def update_interest(self, state, idx):
        """更新用户对样本的兴趣标签"""
        if state == 2:
            self.results[idx] = 1
        else:
            self.results[idx] = 0

    def get_featureNum_and_Map(self):
        """获取频繁模式中出现的特征数和特征对应的特征向量位置"""
        unique_letters = set()

        # 遍历二维列表，将所有字母加入集合中
        for row in self.prevalent_patterns:
            for char in row:
                unique_letters.add(char)  # 集合自动去重

        self.featureNum = len(unique_letters)

        # 将unique_letters中的不重复特征按照字母顺序进行排序
        sorted_letters = sorted(unique_letters)

        # 建立映射表
        index = 0
        self.map = {}
        for feature in sorted_letters:
            self.map[feature] = index
            index += 1

    def get_user_selection(self):
        """获取用户的选择，并在对话框中显示结果。"""
        # 将样本和标签加入到已标记样本集中
        for pattern in self.samples:
            self.patterns.append(pattern)
        for result in self.results:
            self.labels.append(result)

        # 添加执行信息
        all_text = self.progressText.toPlainText()
        # 将文本按行分割，并获取最后一行
        last_line = all_text.splitlines()[-1] if all_text else ""
        words = last_line.split()
        self.progressText.append(words[2] + " interaction completed.")

        # 清空样本、标签
        self.samples = []
        self.results = []

        # 利用已标记样本集训练svm分类器
        self.train()
        # 利用训练好的svm选择新的样本更新已标记样本集
        self.select_samples()

        if len(self.samples) > 0:
            # 添加执行信息
            self.interact_times += 1
            if self.interact_times == 2:
                self.progressText.append("Executing the 2nd interaction...")
            elif self.interact_times == 3:
                self.progressText.append("Executing the 3rd interaction...")
            else:
                self.progressText.append("Executing the {}th interaction...".format(self.interact_times))

            # 利用训练好的svm分类器预测所有频繁模式，显示出用户可能感兴趣的模式
            x_prevalent = patterns2code(self.sorted_prevalent_patterns, self.featureNum, self.map)
            interested_pattern = ""
            for index, x in enumerate(x_prevalent):
                if self.classifier.predict([x]) == [1]:
                    interested_pattern += "{"
                    for i in range(len(self.sorted_prevalent_patterns[index])):
                        if i != len(self.sorted_prevalent_patterns[index]) - 1:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]] + ', '
                        else:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]]
                    interested_pattern += "} "
            self.patternText.setText(interested_pattern)
            font = QFont()
            font.setPointSize(12)  # 设置字体大小为 12
            self.patternText.setFont(font)

            # 弹出交互界面
            self.show_interaction_window()
        else:
            # 利用训练好的svm分类器预测所有频繁模式，显示出用户可能感兴趣的模式
            x_prevalent = patterns2code(self.sorted_prevalent_patterns, self.featureNum, self.map)
            interested_pattern = ""
            for index, x in enumerate(x_prevalent):
                if self.classifier.predict([x]) == [1]:
                    interested_pattern += "{"
                    for i in range(len(self.sorted_prevalent_patterns[index])):
                        if i != len(self.sorted_prevalent_patterns[index]) - 1:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]] + ', '
                        else:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]]
                    interested_pattern += "} "
            self.patternText.setText(interested_pattern)
            font = QFont()
            font.setPointSize(12)  # 设置字体大小为 18
            self.patternText.setFont(font)

    def train(self):
        """利用已标记样本集训练svm分类器"""
        # 对模式进行编码
        x_train = patterns2code(self.patterns, self.featureNum, self.map)
        y_train = self.labels

        # 训练
        if len(x_train) == len(y_train):
            self.classifier.fit(x_train, y_train)

    def select_samples(self):
        """利用训练好的svm选择新的样本更新已标记样本集"""
        x_prevalent = patterns2code(self.prevalent_patterns, self.featureNum, self.map)
        # 使用 predict_proba() 获取候选频繁模式数据的类别概率
        probabilities = self.classifier.predict_proba(x_prevalent)

        # 计算每个样本的信息熵，并保存到列表中
        entropies = [calculate_entropy(prob) for prob in probabilities]

        # 每次采样信息熵的阈值都减少，增强条件
        self.entropy_threshold = self.entropy_threshold - 0.05

        top_entropies = []
        for index, entropy in enumerate(entropies):
            # 选择大于阈值的信息熵和对应的索引
            if entropy >= self.entropy_threshold:
                top_entropies.append([index, entropy])

        # 选择样本
        if len(top_entropies) >= 15:
            # 选择熵值最大的前20个样本
            temp_entropies = sorted(top_entropies, key=lambda x: x[1], reverse=True)
            for item in temp_entropies:
                if len(self.samples) < 15:
                    self.samples.append(self.prevalent_patterns[item[0]])
                else:
                    break
        else:
            for item in top_entropies:
                self.samples.append(self.prevalent_patterns[item[0]])


def patterns2code(patterns, featureNum, location_map):
    """将接收到的模式转换为one-hot编码"""
    x_train = []

    for pattern in patterns:
        x = [0.0] * featureNum
        for feature in pattern:
            x[location_map[feature]] = 1.0
        x_train.append(x)

    return x_train


def calculate_entropy(prob):
    """计算单个样本信息熵的函数"""
    return -sum(p * math.log2(p) for p in prob if p > 0)


if __name__ == '__main__':
    # 创建应用程序实例
    app = QApplication(sys.argv)
    # 创建窗口实例
    demo = Demo()
    # 显示窗口
    demo.show()
    # 运行应用程序
    sys.exit(app.exec_())
