import random
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# 捏特征
letter_to_feature = {'A': 'Supermarket', 'B': 'Milk Tea Shop', 'C': 'Pharmacy', 'D': 'Restaurant',
                     'E': 'Clothing Store', 'F': 'Antique Shop', 'G': 'Barber Shop', 'H': 'Electronic Store',
                     'I': 'Gift Shop', 'J': 'Theme Park', 'K': 'Fruit Shop', 'L': 'Cafe', 'M': 'Optical Shop',
                     'N': 'Dessert Shop', 'O': 'Movie Theater', 'P': 'Beauty Salon', 'Q': 'Bookstore',
                     'R': 'Hotel', 'S': 'Flower Shop', 'T': 'Gym'}


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

        # 调整右边距，为图例留出足够的空间
        fig.subplots_adjust(right=0.80)

        # 初始化空白的二维坐标系
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_title("2D Data Distribution Chart")
        self.ax.grid(True)

    def plot_points(self, points):
        self.ax.clear()  # 清空当前图形

        # 重新绘制坐标系和网格
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_title("2D Data Distribution Chart")
        self.ax.grid(True)

        # 定义可用的几何图形和颜色
        base_shapes = ['o', 's', '^', 'D', 'P', '*', 'X']  # 圆形、方形、三角形、菱形、五边形、星形、X形
        base_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 蓝、绿、红、青、紫、黄、黑

        # 获取数据中的特征类型并建立特征映射表
        features = sorted(set(data[0] for data in points))  # 获取唯一的特征类型，并排序以保持一致性
        shapes = (base_shapes * ((len(features) // len(base_shapes)) + 1))[:len(features)]
        colors = (base_colors * ((len(features) // len(base_colors)) + 1))[:len(features)]
        random.shuffle(shapes)  # 随机排列形状列表
        random.shuffle(colors)  # 随机排列颜色列表

        # 建立特征到形状和颜色的映射表
        feature_style_map = {feature: (shapes[i], colors[i]) for i, feature in enumerate(features)}

        # 绘制每个点
        for instance in points:
            feature = instance[0]
            shape, color = feature_style_map[feature]
            # `instance[2]` 和 `instance[3]` 是坐标点
            self.ax.scatter(instance[2], instance[3], marker=shape, color=color, s=100,
                            label=letter_to_feature[feature])

        # 设置坐标轴标签和标题
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_title('2D Data Distribution Chart')

        # 控制图例，使每个特征只出现一次
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5),
                       title="features", fontsize=12)

        self.draw()


# class MainWindow(QWidget):
#     def __init__(self, points):
#         super().__init__()
#         self.points = points  # 存储传入的二维列表
#         self.initUI()
#
#     def initUI(self):
#         self.setWindowTitle("Data Distribution")
#         self.resize(1600, 900)  # 增加窗口宽度
#
#         # 创建垂直布局并将其应用到主窗口
#         layout = QVBoxLayout(self)
#
#         # 创建绘图区域
#         self.canvas = PlotCanvas(self, width=16, height=12)
#         layout.addWidget(self.canvas)
#
#         # 绘制传入的点
#         self.canvas.plot_points(self.points)


# 主程序示例
# if __name__ == '__main__':
#     data = []
#     with open("data.csv", 'r', encoding='utf-8') as csvfile:
#         csvreader = csv.reader(csvfile)
#         for row in csvreader:
#             if len(row[0]) > 1:
#                 continue
#             temp = [row[0], int(row[1]), float(row[2]), float(row[3])]
#             data.append(temp)
#     del data[0]
#
#     print(data)
#
#     app = QApplication(sys.argv)
#     window = MainWindow(data)
#     window.show()
#     sys.exit(app.exec_())
