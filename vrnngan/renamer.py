import os

def rename_svc_files():
    # Define the target directory containing .svc files
    target_directory = os.path.join(os.path.dirname(__file__), '1-45', 'All')

    # Check if the target directory exists
    if not os.path.exists(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
        return

    # Iterate through all files in the target directory
    for filename in os.listdir(target_directory):
        if filename.endswith('.svc'):
            # Define the new filename by prepending 'collection1'
            new_filename = f"collection1{filename}"
            
            # Construct full paths for the old and new filenames
            old_file_path = os.path.join(target_directory, filename)
            new_file_path = os.path.join(target_directory, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    rename_svc_files()
