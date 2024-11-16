import os
import pandas as pd
import matplotlib.pyplot as plt

def visualize_handwriting_from_file(file_path, ax, title):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        # Open the file and read lines
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # First line contains the sequence length, we can ignore it for plotting
        seq_len = int(lines[0].strip())  # First line is the sequence length
        
        # Parse the remaining data
        data = []
        for line in lines[1:]:  # Skip the first line
            # Split the line into columns and handle any irregular spaces
            columns = line.split()
            if len(columns) == 7:  # Make sure the line contains exactly 7 columns
                data.append(list(map(float, columns)))

        # Convert the list of data into a DataFrame
        df = pd.DataFrame(data, columns=['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude'])
        
        # Separate strokes based on pen status
        on_surface = df[df['pen_status'] == 1]
        in_air = df[df['pen_status'] == 0]

        # Plot on the given axis (ax)
        ax.scatter(on_surface['y'], on_surface['x'], c='b', s=1, alpha=0.7, label='On Surface')
        ax.scatter(in_air['y'], in_air['x'], c='r', s=1, alpha=0.7, label='In Air')
        ax.set_title(title)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.legend()

    except Exception as e:
        print(f"Could not process file {file_path}: {e}")

def compare_handwriting_files(file_path1, file_path2):
    # Create a figure with 2 subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first file in the first subplot
    visualize_handwriting_from_file(file_path1, axes[0], "Handwriting Sample 1")
    
    # Plot the second file in the second subplot
    visualize_handwriting_from_file(file_path2, axes[1], "Handwriting Sample 2")
    
    # Show the side-by-side comparison plot
    plt.tight_layout()
    plt.show()

# Usage Example:
# Specify the paths to the two handwriting files for comparison
file_path1 = './TimeGan/batch/collection1u00001s00001_hw00006.svc'  # Change to your actual file path 1
file_path2 = './TimeGan/output/synthetic_collection1u00001s00001_hw00006.svc'  # Change to your actual file path 2

compare_handwriting_files(file_path1, file_path2)
