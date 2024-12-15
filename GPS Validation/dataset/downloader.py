import os
import requests

def download_tenv3_file(url, save_folder):
    try:
        # Send a GET request to fetch the file
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Extract the file name from the URL (the last part of the URL)
            file_name = url.split("/")[-1]
            
            # Create the full path to save the file in the specified folder
            save_path = os.path.join(save_folder, file_name)
            
            # Ensure the directory exists
            os.makedirs(save_folder, exist_ok=True)

            # Write the content of the response to a file
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"File downloaded successfully and saved as {save_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
url = "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/YMER.tenv3"
save_folder = r"GPS Time Series Model (Validation Tool)\dataset"  # Folder where the file will be saved

download_tenv3_file(url, save_folder)