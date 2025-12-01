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

from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn import svm
import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F

from joinbased import get_prevalent_patterns
import data_distribution
import float_input_dialog
import document
from interaction_window import CoLocationPatternWidget
from prototypical_network import PrototypicalNetwork


letter_to_feature = {'A': 'Supermarket', 'B': 'Milk Tea Shop', 'C': 'Pharmacy', 'D': 'Restaurant',
                     'E': 'Clothing Store', 'F': 'Antique Shop', 'G': 'Barber Shop', 'H': 'Electronic Store',
                     'I': 'Gift Shop', 'J': 'Theme Park', 'K': 'Fruit Shop', 'L': 'Cafe', 'M': 'Optical Shop',
                     'N': 'Dessert Shop', 'O': 'Movie Theater', 'P': 'Beauty Salon', 'Q': 'Bookstore',
                     'R': 'Hotel', 'S': 'Flower Shop', 'T': 'Gym'}


def patterns2code(patterns, featureNum, location_map):
    """
    Convert the received patterns into one-hot encoded vectors.

    This function transforms each input pattern (a list of features) into a
    one-hot encoded vector of length `featureNum`. For every feature appearing
    in the pattern, the corresponding index in the vector is set to 1.0 based
    on the mapping provided in `location_map`.

    Args:
        patterns (list of lists): A list where each element is a pattern represented
                                  by a list of feature identifiers.
        featureNum (int): The total number of possible features, determining the
                          length of each one-hot encoded vector.
        location_map (dict): A dictionary mapping each feature to its index position
                             in the resulting one-hot encoded vector.

    Returns:
        list of lists: A list of one-hot encoded vectors corresponding to the input patterns.
    """
    x_train = []

    for pattern in patterns:
        x = [0.0] * featureNum
        for feature in pattern:
            x[location_map[feature]] = 1.0
        x_train.append(x)

    return x_train


def calculate_entropy(prob):
    """
    Compute the information entropy of a single sample.

    This function calculates the entropy based on a probability distribution
    provided as a list. Only positive probability values contribute to the
    calculation to avoid undefined log operations.

    Args:
        prob (list or iterable): A list of probability values for a sample.

    Returns:
        float: The computed information entropy.
    """
    return -sum(p * math.log2(p) for p in prob if p > 0)


class WorkerThread(QThread):
    # Signal emitted when the background task is completed.
    # It carries the result (a list) back to the main thread.
    task_completed = pyqtSignal(list)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename  # Store the input parameter for the background task

    def run(self):
        """
        Execute the long-running task in a separate thread.

        This method is automatically called when the thread starts.
        It processes the input file by calling get_prevalent_patterns(),
        then emits the result through the task_completed signal.
        """
        result = get_prevalent_patterns(self.filename)  # 调用耗时任务，传入参数
        self.task_completed.emit(result)  # 任务完成后发出信号，传递返回值


def sample_and_remove(data, k):
    """
    Randomly select k sublists from a 2D list, remove them from the original list,
    and return the selected sublists.

    Args:
        data (list of lists): The original 2D list.
        k (int): Number of sublists to sample.

    Returns:
        list of lists: The sampled sublists.
    """
    assert k > 0, "k must be a positive integer"
    assert k <= len(data), "k cannot exceed the number of available sublists"

    # Randomly select k unique indices
    selected_indices = random.sample(range(len(data)), k)

    # Get the selected sublists
    selected = [data[i] for i in selected_indices]

    # Delete selected items from data (sort indices to avoid shifting problems)
    for idx in sorted(selected_indices, reverse=True):
        del data[idx]

    return selected


def split_patterns_with_labels(patterns, labels, test_size=0.25, random_state=None):
    """
    Split patterns and their corresponding labels into training and testing sets.

    Args:
        patterns (list): 2D list, each sublist represents a pattern.
        labels (list): 1D list, each element represents the label of the corresponding pattern.
        test_size (float): Proportion of the test set, default is 0.25 (i.e., 3:1 ratio).
        random_state (int): Random seed for reproducible results.

    Returns:
        train_patterns, test_patterns, train_labels, test_labels
    """
    # Convert to numpy arrays for processing
    patterns_array = np.array(patterns, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int64)

    # Use sklearn's train_test_split to split the data
    train_pat, test_pat, train_lab, test_lab = train_test_split(
        patterns_array,
        labels_array,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array  # Preserve label distribution
    )

    # Convert back to list form for output
    return train_pat, test_pat, train_lab, test_lab


class Demo(QMainWindow):
    """Main interface for the ICPAL demo application."""

    def __init__(self):
        super().__init__()

        # Path to the loaded data file
        self.filename = None

        # List to store prevalent patterns extracted from the dataset
        self.prevalent_patterns = None

        # Prevalent patterns sorted by increasing pattern size
        self.sorted_prevalent_patterns = None

        # Sample set selected during each interaction round
        self.samples = []

        # Corresponding labels assigned by the user for each selected sample
        self.results = []

        # Labeled dataset used for training
        self.patterns = []  # List of pattern feature vectors
        self.labels = []  # Corresponding interest labels

        # Number of features
        self.featureNum = 0

        # Mapping from features to their positions in the feature vector
        self.map = {}

        # Classifier used for prediction during interactive learning
        self.classifier = None

        # Default entropy threshold
        self.entropy_threshold = 0.9

        # Window that displays documentation or help content
        self.documentation_window = None

        # Counter for the number of interaction rounds
        self.interact_times = 0

        self.prototypes = None

        # Initialize all UI components
        self.initUI()

    def initUI(self):
        """Initialize and configure the user interface."""

        # Set window title
        self.setWindowTitle('ICPAL')

        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()

        # Resize window to match screen resolution
        self.setGeometry(0, 0, screen_width, screen_height)

        # Set application icon
        self.setWindowIcon(QIcon("icons/icon.png"))

        # Create toolbar
        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(toolbar)

        # Action: open data file
        open_file_action = QAction(QIcon("icons/file_icon.png"), "Open File", self)
        open_file_action.triggered.connect(self.on_view_click)
        toolbar.addAction(open_file_action)

        # Add spacing between toolbar items
        spacer1 = QWidget()
        spacer1.setFixedWidth(50)
        toolbar.addWidget(spacer1)

        # Action: set information entropy threshold
        threshold_action = QAction(QIcon("icons/threshold_icon.png"), "Set Information Entropy Threshold", self)
        threshold_action.triggered.connect(self.set_threshold)
        toolbar.addAction(threshold_action)

        # Add spacing
        spacer2 = QWidget()
        spacer2.setFixedWidth(50)
        toolbar.addWidget(spacer2)

        # Action: start mining algorithm
        start_action = QAction(QIcon("icons/start_icon.png"), "Start Mining", self)
        start_action.triggered.connect(self.start_click)
        toolbar.addAction(start_action)

        # Add spacing
        spacer3 = QWidget()
        spacer3.setFixedWidth(50)
        toolbar.addWidget(spacer3)

        # Action: open documentation window
        about_action = QAction(QIcon("icons/document_icon.png"), "Documentation", self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)

        # Central widget and main layout
        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Container for user interaction and data distribution visualization
        self.data_and_distribution_widget = QWidget()
        self.data_and_distribution_layout = QHBoxLayout()
        self.data_and_distribution_widget.setLayout(self.data_and_distribution_layout)

        # Table to display raw data (patterns or pattern details)
        self.table_widget = QTableWidget()
        self.table_widget.setFixedSize(int(screen_width * 0.3), int(screen_height * 0.6))
        self.data_and_distribution_layout.addWidget(self.table_widget)

        # Set table resizing behavior
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Widget for showing data distribution plot
        self.data_widget = QWidget()
        self.data_widget.setFixedSize(int(screen_width * 0.6), int(screen_height * 0.6))
        self.data_layout = QHBoxLayout()
        self.data_widget.setLayout(self.data_layout)
        # Customized Matplotlib canvas showing pattern distribution
        self.canvas = data_distribution.PlotCanvas(self, width=17, height=9)
        self.data_layout.addWidget(self.canvas)
        self.data_and_distribution_layout.addWidget(self.data_widget)
        self.layout.addWidget(self.data_and_distribution_widget)

        # Tabs: progress, prevalent patterns, user interested patterns
        self.tab_widget = QTabWidget()
        self.tab_widget.setFixedSize(screen_width, int(screen_height * 0.25))
        # Progress text area
        self.progressText = QTextEdit(self)
        self.progressText.setReadOnly(True)
        font = QFont()
        font.setPointSize(12)  # 设置字体大小为 12
        self.progressText.setFont(font)
        self.tab_widget.addTab(self.progressText, "Progress")
        # Prevalent patterns text area
        self.prevalentText = QTextEdit(self)
        self.prevalentText.setReadOnly(True)
        self.tab_widget.addTab(self.prevalentText, "Prevalent")
        # Results: patterns predicted to be user-interesting
        self.patternText = QTextEdit(self)
        self.patternText.setReadOnly(True)
        self.tab_widget.addTab(self.patternText, "Results")
        self.layout.addWidget(self.tab_widget)

        # Bottom status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Label showing the current task name
        self.task_label = QLabel("Generating frequent patterns...")
        self.task_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.task_label.setVisible(False)

        # Indeterminate progress bar for long-running tasks
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignRight)
        self.progress_bar.setFixedSize(200, 20)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode

        # Add widgets to status bar
        self.status_bar.addPermanentWidget(self.task_label)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Set central widget and layout
        self.setCentralWidget(central_widget)
        self.setLayout(self.layout)

        # Launch in maximized mode
        self.showMaximized()

    def show_about(self):
        """Open the documentation/help window."""

        # Create the documentation window if it has not been initialized yet
        if self.documentation_window is None:
            self.documentation_window = document.DocumentationWindow()

        # Display the documentation window
        self.documentation_window.show()

    def set_threshold(self):
        """Set the information entropy threshold."""

        # Pop up an input dialog to allow the user to enter a floating-point value
        while True:
            # Create and display the custom dialog
            dialog = float_input_dialog.FloatInputDialog()

            if dialog.exec_() == QDialog.Accepted:
                # Retrieve the input value and validate its range
                try:
                    value = float(dialog.get_value())

                    if 0 <= value <= 1:
                        QMessageBox.information(
                            self,
                            "Succeed",
                            f"Successfully set the information entropy threshold to {value}."
                        )
                        self.entropy_threshold = value
                        break
                    else:
                        QMessageBox.warning(
                            self,
                            "Error",
                            "Please enter a floating-point number between 0 and 1!"
                        )
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Error",
                        "Please enter a valid floating-point number!"
                    )
            else:
                # User cancelled the dialog
                break

    def on_view_click(self):
        """Select and load a data file."""

        # Open a file selection dialog
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )

        # Save the selected file path
        self.filename = fileName

        # Display execution information
        self.progressText.setText("Successfully opened the file " + fileName)

        # Read and preprocess the data
        data = []
        with open(self.filename, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                # Skip invalid rows (e.g., feature name length > 1)
                if len(row[0]) > 1:
                    continue
                # Convert fields: [feature, int, float, float]
                temp = [row[0], int(row[1]), float(row[2]), float(row[3])]
                data.append(temp)
        del data[0]  # Remove header row

        # Read and display the CSV file content in the table widget
        with open(fileName, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            row = list(reader)  # Convert CSV content to a list

            # Set the number of rows and columns in the table
            self.table_widget.setRowCount(len(row))
            self.table_widget.setColumnCount(len(row[0]))

            # Insert data into the GUI table
            for row_index, row_data in enumerate(row):
                # Convert feature letters for all rows except the header
                if row_index != 0:
                    row_data[0] = letter_to_feature[row_data[0]]
                for col_index, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(cell_data)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make cells read-only
                    item.setTextAlignment(Qt.AlignCenter)  # Center-align text
                    self.table_widget.setItem(row_index, col_index, item)

        # Visualize the loaded data points
        self.canvas.plot_points(data)

    def start_click(self):
        """Start executing the program"""
        # Show task label and progress bar
        self.task_label.setVisible(True)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()  # Force UI update

        # Add execution information
        self.progressText.append("Generating frequent patterns...")

        # Start the background thread and pass parameters to the thread
        self.worker_thread = WorkerThread(self.filename)
        self.worker_thread.task_completed.connect(self.handle_task_result)  # Connect signal to result handling function
        # Record the start time of frequent pattern generation
        self.start_time = time.time()
        self.worker_thread.start()

    def handle_task_result(self, result):
        """Continue executing the algorithm"""
        # Record the time when frequent pattern generation is completed
        self.end_time = time.time()
        # Store the frequent patterns
        self.prevalent_patterns = result
        self.sorted_prevalent_patterns = sorted(result, key=lambda x: len(x))

        # Add execution information
        self.progressText.append("Successfully generated frequent patterns, taking {:.2f} seconds."
                                 .format(self.end_time - self.start_time))

        # Display the frequent patterns
        text = ""
        for prevalent_pattern in self.sorted_prevalent_patterns:
            text += "{"
            for index in range(len(prevalent_pattern)):
                if index == len(prevalent_pattern) - 1:
                    text += letter_to_feature[prevalent_pattern[index]] + "} "
                else:
                    text += letter_to_feature[prevalent_pattern[index]] + ", "
        # Set font size
        font = QFont()
        font.setPointSize(12)  # Set font size to 12
        self.prevalentText.setFont(font)
        self.prevalentText.setText(text)
        QApplication.processEvents()  # Force UI update

        # Hide progress bar and task label when task is completed, and show completion message
        self.task_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Done.", 3000)  # Show completion message for 3 seconds

        # Get the number of features appearing in frequent patterns and the mapping to feature vector positions
        self.get_featureNum_and_Map()
        self.classifier = PrototypicalNetwork(input_size=self.featureNum, embedding_size=64, num_classes=2)
        # Shuffle the frequent patterns randomly
        random.shuffle(self.prevalent_patterns)

        # For the first execution, randomly sample a part of patterns as the training set
        k = 10
        self.samples = sample_and_remove(self.prevalent_patterns, k)

        # Add execution information
        self.interact_times += 1
        self.progressText.append("Executing the 1st interaction...")

        # Pop up the interaction window
        self.show_interaction_window()

    def show_interaction_window(self):
        """Pop up the interaction window"""
        self.interaction_window = CoLocationPatternWidget(self.samples)
        self.interaction_window.interestSubmitted.connect(self.get_interest_labels)
        self.interaction_window.show()

    def get_interest_labels(self, interest_labels):
        """Obtain interaction results"""
        self.results = interest_labels
        self.get_user_selection()

    def update_interest(self, state, idx):
        """Update the user's interest label for a sample"""
        if state == 2:
            self.results[idx] = 1
        else:
            self.results[idx] = 0

    def get_featureNum_and_Map(self):
        """Get the number of features appearing in frequent patterns and their positions in the feature vector"""
        unique_letters = set()

        # Traverse the 2D list and add all letters to the set
        for row in self.prevalent_patterns:
            for char in row:
                unique_letters.add(char)  # Set automatically removes duplicates

        self.featureNum = len(unique_letters)

        # Sort the unique features in alphabetical order
        sorted_letters = sorted(unique_letters)

        # Create a mapping table
        index = 0
        self.map = {}
        for feature in sorted_letters:
            self.map[feature] = index
            index += 1

    def get_user_selection(self):
        """Obtain the user's selection and display the results in the dialog."""
        # Add the samples and labels to the labeled sample set
        for pattern in self.samples:
            self.patterns.append(pattern)
        for result in self.results:
            self.labels.append(result)

        # Add execution information
        all_text = self.progressText.toPlainText()
        # Split the text by lines and get the last line
        last_line = all_text.splitlines()[-1] if all_text else ""
        words = last_line.split()
        self.progressText.append(words[2] + " interaction completed.")

        # Clear samples and labels
        self.samples = []
        self.results = []

        # Train classifier with the labeled sample set
        self.train()
        # Use the trained classifier to select new samples and update the labeled sample set
        self.select_samples()

        if len(self.samples) > 0:
            # Add execution information
            self.interact_times += 1
            if self.interact_times == 2:
                self.progressText.append("Executing the 2nd interaction...")
            elif self.interact_times == 3:
                self.progressText.append("Executing the 3rd interaction...")
            else:
                self.progressText.append("Executing the {}th interaction...".format(self.interact_times))

            # Use the trained classifier to predict all frequent patterns and display patterns the user may be interested in
            x_prevalent = patterns2code(self.sorted_prevalent_patterns, self.featureNum, self.map)
            x_embeddings = self.classifier(torch.tensor(x_prevalent, dtype=torch.float32), None, None, None, 'cal_only')
            distances = torch.cdist(x_embeddings, self.prototypes)
            neg_distances = -distances
            pattern_probabilities = F.softmax(neg_distances, dim=1)
            interested_pattern = ""
            for index, x in enumerate(x_prevalent):
                if pattern_probabilities[index][1] >= 0.5:
                    interested_pattern += "{"
                    for i in range(len(self.sorted_prevalent_patterns[index])):
                        if i != len(self.sorted_prevalent_patterns[index]) - 1:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]] + ', '
                        else:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]]
                    interested_pattern += "} "
            self.patternText.setText(interested_pattern)
            font = QFont()
            font.setPointSize(12)  # Set font size to 12
            self.patternText.setFont(font)

            # Pop up the interaction window
            self.show_interaction_window()
        else:
            # Use the trained SVM classifier to predict all frequent patterns and display patterns the user may be interested in
            x_prevalent = patterns2code(self.sorted_prevalent_patterns, self.featureNum, self.map)
            x_embeddings = self.classifier(torch.tensor(x_prevalent, dtype=torch.float32), None, None, None, 'cal_only')
            distances = torch.cdist(x_embeddings, self.prototypes)
            neg_distances = -distances
            pattern_probabilities = F.softmax(neg_distances, dim=1)
            interested_pattern = ""
            for index, x in enumerate(x_prevalent):
                if pattern_probabilities[index][1] >= 0.5:
                    interested_pattern += "{"
                    for i in range(len(self.sorted_prevalent_patterns[index])):
                        if i != len(self.sorted_prevalent_patterns[index]) - 1:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]] + ', '
                        else:
                            interested_pattern += letter_to_feature[self.sorted_prevalent_patterns[index][i]]
                    interested_pattern += "} "
            self.patternText.setText(interested_pattern)
            font = QFont()
            font.setPointSize(12)  # Set font size to 12
            self.patternText.setFont(font)

    def train(self):
        """Train the classifier using the labeled sample set"""
        # Encode the patterns
        x_train = patterns2code(self.patterns, self.featureNum, self.map)
        y_train = self.labels

        train_patterns, test_patterns, train_labels, test_labels = split_patterns_with_labels(x_train,
                                                                                              y_train)

        # Train the classifier
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.0001)
        epochs = len(x_train)
        for i in range(epochs):
            optimizer.zero_grad()
            loss, self.prototypes = self.classifier(torch.from_numpy(train_patterns), torch.from_numpy(train_labels),
                                                    torch.from_numpy(test_patterns), torch.from_numpy(test_labels),
                                                    'train')
            loss.backward()
            optimizer.step()


    def select_samples(self):
        """Use the trained SVM to select new samples and update the labeled sample set"""
        x_prevalent = patterns2code(self.prevalent_patterns, self.featureNum, self.map)
        x_embeddings = self.classifier(torch.tensor(x_prevalent, dtype=torch.float32), None, None, None, 'cal_only')
        distances = torch.cdist(x_embeddings, self.prototypes)
        neg_distances = -distances
        pattern_probabilities = F.softmax(neg_distances, dim=1)

        # Calculate the information entropy for each sample and save it in a list
        entropies = [calculate_entropy(prob) for prob in pattern_probabilities]

        # Reduce the entropy threshold each time to increase selection stringency
        self.entropy_threshold = self.entropy_threshold - 0.05

        top_entropies = []
        for index, entropy in enumerate(entropies):
            # Select entropies and corresponding indices that exceed the threshold
            if entropy >= self.entropy_threshold:
                top_entropies.append([index, entropy])

        # Select samples
        if len(top_entropies) >= 15:
            # Sort by entropy value
            temp_entropies = sorted(top_entropies, key=lambda x: x[1], reverse=True)
            selected = temp_entropies[:15]  # Take the top 15

        elif len(top_entropies) > 0:
            selected = top_entropies[:]  # Select all

        else:
            selected = []

        # Add to samples
        for item in selected:
            self.samples.append(self.prevalent_patterns[item[0]])

        # --- Remove selected samples from the original list ---
        # Extract the indices of selected samples
        selected_indices = [item[0] for item in selected]

        # Delete by index from largest to smallest to avoid misalignment
        for idx in sorted(selected_indices, reverse=True):
            del self.prevalent_patterns[idx]


if __name__ == '__main__':
    # Create the application instance
    app = QApplication(sys.argv)
    # Create the window instance
    demo = Demo()
    # Show the window
    demo.show()
    # Run the application
    sys.exit(app.exec_())

