from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QToolBar, QAction
import sys


class DocumentationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document")
        self.resize(1000, 800)  # Set window size

        # Set the window icon
        self.setWindowIcon(QIcon("icons/document_window_icon.png"))

        # Create a QTextEdit to display the document content
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)  # Set to read-only mode

        # Document content
        documentation_text = """
Software User Guide：

1. Open File: Click "Open File" in the toolbar to browse and select the CSV data file you want to process.

2. Set Parameters: Click "Set Information Entropy Threshold" in the toolbar to enter a floating-point number between 0 and 1 as the entropy threshold for the algorithm.

3. Start Algorithm: Click "Start Mining" in the toolbar to begin the algorithm. The system will first generate frequent patterns, then interact with you through an interactive window. If you are interested in a pattern, you can check the box next to it; if not, you can leave it unchecked. After the interaction, the patterns you are interested in will be displayed at the bottom of the system interface.

4. The display area at the bottom of the interface includes the following sections: the “Progress” section shows task status messages during system execution; the “Prevalent” section displays all frequent patterns after mining is complete; and the “Results” section shows patterns that may be of interest to you after each interaction.
        """

        # Load the document content into QTextEdit
        self.text_edit.setText(documentation_text)
        font = QFont()
        font.setPointSize(12)  # Set font size to 12
        self.text_edit.setFont(font)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)

        # Apply the layout to the window
        self.setLayout(layout)
