# An Interactive Co-location Pattern Mining System Based on Active Learning

## Overview
This repository contains the code and resources for a demonstration project designed to showcase various functionalities such as spatial data distribution visualization, frequent pattern mining, and interactive user interfaces.

## Repository Structure

### Folders
- **data/**
  - Contains test dataset used for the demo.

- **icons/**
  - Contains icons used in the demo interface.

### Files

1. **data_distribution.py**
   - Visualizes the distribution of spatial data selected by the user.

2. **demo.py**
   - The main program of the demo.

3. **document.py**
   - Implements the documentation window for the demo.

4. **float_input_dialog.py**
   - Implements the parameter setting window for the demo.

5. **interaction_window.py**
   - Implements the human-computer interaction interface of the demo.

6. **joinbased.py**
   - Implements prevalent pattern mining from the dataset.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python libraries (can be installed via `requirements.txt` if provided, or specify them here).

### Setup
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:
   ```bash
   cd ICPAL
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the `data/` folder contains the necessary datasets and the `icons/` folder contains the required icons.

### Running the Demo
Execute the main program:
```bash
python demo.py
```
