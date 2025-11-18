import random
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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

        # Adjust right margin to leave enough space for the legend
        fig.subplots_adjust(right=0.80)

        # Initialize a blank 2D coordinate system
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_title("2D Data Distribution Chart")
        self.ax.grid(True)

    def plot_points(self, points):
        self.ax.clear()  # Clear the current figure

        # Redraw axes and grid
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_title("2D Data Distribution Chart")
        self.ax.grid(True)

        # Define available shapes and colors
        base_shapes = ['o', 's', '^', 'D', 'P', '*', 'X']  # Circle, square, triangle, diamond, pentagon, star, X
        base_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Blue, green, red, cyan, magenta, yellow, black

        # Get feature types from the data and create a feature mapping table
        features = sorted(set(data[0] for data in points))  # Get unique feature types and sort to maintain consistency
        shapes = (base_shapes * ((len(features) // len(base_shapes)) + 1))[:len(features)]
        colors = (base_colors * ((len(features) // len(base_colors)) + 1))[:len(features)]
        random.shuffle(shapes)  # Randomly shuffle the shapes list
        random.shuffle(colors)  # Randomly shuffle the colors list

        # Create a mapping table from feature to shape and color
        feature_style_map = {feature: (shapes[i], colors[i]) for i, feature in enumerate(features)}

        # Plot each point
        for instance in points:
            feature = instance[0]
            shape, color = feature_style_map[feature]
            # `instance[2]` and `instance[3]` are the coordinates
            self.ax.scatter(instance[2], instance[3], marker=shape, color=color, s=100,
                            label=letter_to_feature[feature])

        # Set axis labels and title
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_title('2D Data Distribution Chart')

        # Control the legend so that each feature appears only once
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5),
                       title="features", fontsize=12)

        self.draw()
