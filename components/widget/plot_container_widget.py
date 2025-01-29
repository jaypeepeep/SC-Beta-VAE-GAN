from PyQt5 import QtWidgets, QtCore
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PlotContainerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotContainerWidget, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        # Add a layout for the plots
        self.layout = QtWidgets.QVBoxLayout(self)

        # Create a Matplotlib figure and canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Set the styling for the canvas
        self.canvas.setStyleSheet("border: 5px solid #000; background-color: #FFF;")

        # Set a minimum height for the canvas
        self.canvas.setMinimumHeight(600)  # Adjust the height as needed

        # Add the canvas to the layout
        self.layout.addWidget(self.canvas)

    def loadPlot(self, filename):
        """Load and plot data from a .svc file."""
        # Construct the path to the uploads folder
        uploads_folder = os.path.join(os.path.dirname(__file__), "../../files/uploads")
        file_path = os.path.join(uploads_folder, filename)

        # Check if the file exists
        if os.path.isfile(file_path):
            self.plot_data(file_path)
        else:
            print(f"File not found: {filename}")

    def plot_data(self, file_path):
        """Plot data from the .svc file."""
        # Clear the previous plot
        self.figure.clear()

        # Read and process the .svc file
        df = pd.read_csv(file_path, skiprows=1, header=None, delim_whitespace=True)
        df.columns = [
            "x",
            "y",
            "timestamp",
            "pen_status",
            "pressure",
            "azimuth",
            "altitude",
        ]

        # Modify timestamp to start from 0
        df["timestamp"] = (df["timestamp"] - df["timestamp"].min()).round().astype(int)
        df = df.iloc[:, [0, 1, 2, 3]]  # Select x, y, timestamp, and pen_status

        on_paper = df[df["pen_status"] == 1]
        in_air = df[df["pen_status"] == 0]

        # Set figure size for landscape orientation
        self.figure.set_size_inches(12, 6)  # Width, Height in inches

        # Create the plot without any transformations
        ax = self.figure.add_subplot(111)

        # Scatter plot with rotated coordinates
        ax.scatter(
            on_paper["x"], -on_paper["y"], c="blue", s=1, alpha=0.7, label="On Surface"
        )
        ax.scatter(in_air["x"], -in_air["y"], c="red", s=1, alpha=0.7, label="In Air")

        ax.set_title(f"Plot from {os.path.basename(file_path)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_aspect("equal")

        # Set x and y limits to adjust for the rotation
        ax.set_xlim(df["x"].min(), df["x"].max())
        ax.set_ylim(-df["y"].max(), -df["y"].min())

        # Refresh the canvas
        self.canvas.draw()

    # def load_plot_from_figure(self, figure):
       # """Load and display a Matplotlib figure."""
        # Clear the previous plot
        # self.figure.clear()

        # Use the provided figure for this canvas
        # self.canvas.figure = figure

        # Redraw the canvas with the new figure
        # self.canvas.draw()