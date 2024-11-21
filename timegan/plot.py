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

def plot_all_files_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    # Get all .svc files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.svc')]
    if not files:
        print(f"No .svc files found in folder: {folder_path}")
        return

    # Create subplots dynamically based on the number of files
    num_files = len(files)
    fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))
    
    # If only one file, axes is not a list
    if num_files == 1:
        axes = [axes]
    
    for ax, file_name in zip(axes, files):
        file_path = os.path.join(folder_path, file_name)
        visualize_handwriting_from_file(file_path, ax, title=file_name)
    
    plt.tight_layout()
    plt.show()

# Usage Example:
# Uncomment one of the following to use the desired functionality:

# To compare two specific files:
# file_path1 = './timeGan/to_augment/collection1u00001s00001_hw00002.svc'  # Change to your actual file path 1
# file_path2 = './timeGan/augmented/synthetic_collection1u00001s00001_hw00002.svc'  # Change to your actual file path 2
# compare_handwriting_files(file_path1, file_path2)

# To plot all files in a folder:
folder_path = './TimeGan/augmented/'  # Change to your folder path
plot_all_files_in_folder(folder_path)
