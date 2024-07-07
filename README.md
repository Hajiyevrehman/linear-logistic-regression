# Linear and Logistic Regression Project

## Overview

This project implements Linear Regression and Logistic Regression algorithms. It includes source code, datasets, and Jupyter notebooks for running and analyzing these implementations.

## File Structure

project_root
├── README.md  (this file)
├── algorithms/  (source code folder)
├── linear_regression.ipynb
├── logistic_regression.ipynb
├── get_datasets.py  (dataset download script)
└── datasets/  (datasets folder)

## Getting Started

### Prerequisites

- Python 3.9.5 or later
- Jupyter Notebook

### Preparing Datasets

Before running the notebooks, download the required datasets by executing:

python get_datasets.py

This command will download the necessary datasets and place them in the ./datasets directory.

## Running the Project

1. Open Jupyter Notebook:
   jupyter notebook
2. Navigate to and open the following notebooks:
   - linear_regression.ipynb
   - logistic_regression.ipynb
3. Run through the cells in each notebook to see the implementations and analyses.

## Implementation Details

The main components of this project are:

1. algorithms/linear_regression.py: Contains the Linear Regression algorithm implementation
2. algorithms/logistic_regression.py: Contains the Logistic Regression algorithm implementation
3. linear_regression.ipynb: Jupyter notebook for running and visualizing Linear Regression
4. logistic_regression.ipynb: Jupyter notebook for running and visualizing Logistic Regression

## Usage Tips

- Run all code cells in the notebooks sequentially.
- The notebooks include inline questions and areas for plotting results.
- Modify only the parts of the code that are explicitly marked for editing.

## Setting Up Python Environment (Optional)

If you're not using Datahub and encounter package issues:

1. Install required packages:
   pip install -r requirements.txt

2. If issues persist, use a conda environment:
   conda create -n regression_project python=3.9.5
   conda activate regression_project
   pip install -r requirements.txt

## Using Jupyter Notebook

For those new to Jupyter Notebook:

- Notebooks consist of code cells and markdown cells.
- To run a cell, select it and press Ctrl + Enter or click the "Run" button.
- Code cells execute Python code; markdown cells render formatted text.
- Variables and functions defined in one cell are available in subsequent cells.

## Exporting Results

To export a notebook as PDF:
1. Go to File > Download as > PDF via LaTeX (.pdf)
2. Ensure all cells have been executed before exporting.

## Support

If you have any questions or encounter issues, please open an issue in this repository.