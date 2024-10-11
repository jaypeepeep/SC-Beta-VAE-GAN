from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QColor
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_container_widget import FileContainerWidget
from components.widget.file_preview_widget import FilePreviewWidget
from components.widget.model_widget import ModelWidget
from components.widget.process_log_widget import ProcessLogWidget
from components.widget.output_widget import OutputWidget
from components.widget.spin_box_widget import SpinBoxWidget
from components.button.DragDrop_Button import DragDrop_Button
from components.widget.result_preview_widget import SVCpreview
from model import scbetavaegan
import os
import time
import shutil

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope

from PyQt5.QtCore import QThread, pyqtSignal
import traceback

class GenerateDataWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)  # For logging progress
    
    def __init__(self, workplace):
        super().__init__()
        self.workplace = workplace
        self.uploaded_files = workplace.uploaded_files
        
    def run(self):
        try:
            # Move all the generation logic here
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            folder_name = f"SyntheticData_{timestamp}"
            output_dir = os.path.join(os.path.dirname(__file__), '../uploads', folder_name)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for file_path in self.uploaded_files:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    destination_path = os.path.join(output_dir, file_name)
                    shutil.copy(file_path, destination_path)
                    self.progress.emit(f"Copied {file_name} to {destination_path}")

            self.progress.emit("Starting synthetic data generation...")
            
            # Simulate some processing steps with different log levels
            self.progress.emit("Checking input files...")
            time.sleep(1) 
            
            if self.uploaded_files:
                self.progress.emit(f"Found {len(self.uploaded_files)} files to process")
                
                for i, file in enumerate(self.uploaded_files, 1):
                    self.progress.emit(f"Processing file {i}: {os.path.basename(file)}")
                    time.sleep(0.5)
                    
                    # Simulate some potential warnings or errors
                    if i % 3 == 0:
                        self.logger.warning(f"Warning: File {i} may contain inconsistent data")
                    if i % 5 == 0:
                        self.logger.error(f"Error: Could not process some sections in file {i}")
                    
                self.progress.emit("Generating synthetic data...")
                time.sleep(2)
                self.progress.emit("Synthetic data generation completed successfully!")
                
                # Expand output and result sections
                QtCore.QTimer.singleShot(0, lambda: self.collapsible_widget_process_log.toggle_container(True))
                QtCore.QTimer.singleShot(3000, lambda: self.collapsible_widget_output.toggle_container(True))
                QtCore.QTimer.singleShot(4000, lambda: self.collapsible_widget_result.toggle_container(True))
                
            else:
                self.logger.warning("No input files found. Please upload files before generating data.")
                QMessageBox.warning(
                    self,
                    'No Files',
                    "No input files uploaded to generate synthetic data",
                    QMessageBox.Ok
                )
            print(self.uploaded_files)

            self.num_files_to_use = 1
            self.data_frames, self.processed_data, self.scalers, self.avg_data_points, self.input_filenames, self.original_data_frames = scbetavaegan.upload_and_process_files(self.uploaded_files, self.num_files_to_use)

            # # Store the name of the first file for use in Cell 4
            self.input_filename = self.input_filenames[0] if self.input_filenames else 'processed_data'
            print(f"Number of processed files: {len(self.processed_data)}") ##dito sa processed data naka_store
            print(f"Average number of data points: {self.avg_data_points}")

            for self.df_idx in range(len(self.data_frames)):
                self.df = self.data_frames[self.df_idx]  # Using each DataFrame in the list

                # Convert the 'timestamp' column to numeric for calculations (if not already done)
                self.df['timestamp'] = pd.to_numeric(self.df['timestamp'])

                # Sort the DataFrame by timestamp (should already be sorted in the function)
                self.df.sort_values('timestamp', inplace=True)

                # Calculate the differences between consecutive timestamps (optional for gap finding)
                self.df['time_diff'] = self.df['timestamp'].diff()

                # Identify the indices where the time difference is greater than 30,000 milliseconds
                self.gap_indices = self.df.index[self.df['time_diff'] > 8].tolist()

                # Create an empty list to hold the new rows
                self.new_rows = []

                # Fill in the gaps with 70 milliseconds intervals
                for self.idx in self.gap_indices:
                    # Check if the next index is valid
                    if self.idx + 1 < len(self.df):
                        # Get the current and next timestamps
                        self.current_timestamp = self.df.at[self.idx, 'timestamp']
                        self.next_timestamp = self.df.at[self.idx + 1, 'timestamp']

                        # Calculate how many entries we need to fill in
                        self.num_fill_entries = (self.next_timestamp - self.current_timestamp) // 7

                        # Generate the timestamps to fill the gap
                        for self.i in range(1, self.num_fill_entries + 1):
                            self.new_timestamp = self.current_timestamp + i * 7

                            # Create a new row to fill in with NaN for x and y
                            self.new_row = {
                                'x': np.nan,  # Set x to NaN
                                'y': np.nan,  # Set y to NaN
                                'timestamp': self.new_timestamp,
                                'pen_status': 0,        # You can set this to your desired value
                                'azimuth': self.df.at[self.idx, 'azimuth'],   # Use the current azimuth value
                                'altitude': self.df.at[self.idx, 'altitude'], # Use the current altitude value
                                'pressure': self.df.at[self.idx, 'pressure']  # Use the current pressure value
                            }

                            # Append the new row to the list of new rows
                            self.new_rows.append(self.new_row)

                # Create a DataFrame from the new rows
                self.new_rows_df = pd.DataFrame(self.new_rows)

                # Concatenate the original DataFrame with the new rows DataFrame
                self.df = pd.concat([self.df, self.new_rows_df], ignore_index=True)

                # Sort the DataFrame by timestamp to maintain order
                self.df.sort_values('timestamp', inplace=True)

                # Reset index after sorting
                self.df.reset_index(drop=True, inplace=True)


                # Check for NaN entries before interpolation
                if self.df[['x', 'y']].isnull().any().any():
                    self.df[['x', 'y']] = self.df[['x', 'y']].interpolate(method='linear')

                # Drop the 'time_diff' column after processing
                self.df.drop(columns=['time_diff'], inplace=True)

                # Update the processed data
                self.data_frames[self.df_idx] = self.df

            # Update processed data for all DataFrames
            self.processed_data = [np.column_stack((self.scaler.transform(self.df[['x', 'y', 'timestamp']]), self.df['pen_status'].values)) 
                            for self.df, self.scaler in zip(self.data_frames, self.scalers)]
            self.avg_data_points = int(np.mean([self.df.shape[0] for self.df in self.data_frames]))

            self.processed_dataframes = []

            for self.input_filename, self.df in zip(self.input_filenames, self.data_frames):
                # Convert all numeric columns to integers
                self.df[['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']] = self.df[['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']].astype(int)
                
                # Append the processed DataFrame to the list
                self.processed_dataframes.append(self.df)

                print(f"Processed DataFrame for: {self.input_filename}")

            # Use the processed_dataframes directly
            self.data_frames, self.processed_data, self.scalers, self.avg_data_points, self.original_filenames = scbetavaegan.process_dataframes(self.processed_dataframes, self.num_files_to_use)
            print(f"Number of processed files: {len(self.processed_data)}")
            print(f"Average number of data points: {self.avg_data_points}")

            self.latent_dim = 128
            self.beta = 0.0001
            self.learning_rate = 0.001
            self.lambda_shift = 0.5

            self.vae = scbetavaegan.VAE(self.latent_dim, self.beta)
            self.optimizer = scbetavaegan.tf.keras.optimizers.Adam(self.learning_rate)

            # Initialize LSTM discriminator and optimizer
            self.lstm_discriminator = scbetavaegan.LSTMDiscriminator()
            self.lstm_optimizer = scbetavaegan.tf.keras.optimizers.Adam(self.learning_rate)

            self.batch_size = 512
            self.train_datasets = [scbetavaegan.tf.data.Dataset.from_tensor_slices(data).shuffle(10000).batch(self.batch_size) for data in self.processed_data]

            # Set up alternating epochs
            self.vae_epochs = 200
            self.lstm_interval = 50
            self.epochs = 5
            self.visual_per_num_epoch = 5
            self.num_augmented_files = 1

            self.generator_loss_history = []
            self.reconstruction_loss_history = []
            self.kl_loss_history = []
            self.nrmse_history = []

            self.save_dir = "vae_models"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            for self.epoch in range(self.epochs):
                self.generator_loss = 0 
                self.reconstruction_loss_sum = 0
                self.kl_loss_sum = 0
                self.num_batches = sum(len(self.dataset) for self.dataset in self.train_datasets)

                with scbetavaegan.tqdm(total=self.num_batches, desc=f'Epoch {self.epoch+1}/{self.epochs}', unit='batch') as pbar:
                    for dataset in self.train_datasets:
                        for self.batch in dataset:
                            self.use_lstm = self.epoch >= self.vae_epochs and (self.epoch - self.vae_epochs) % self.lstm_interval == 0
                            self.generator_loss_batch, self.reconstruction_loss, self.kl_loss = scbetavaegan.train_vae_step(self.vae, self.batch, self.optimizer, self.lstm_discriminator if self.use_lstm else None)
                            self.generator_loss += self.generator_loss_batch
                            self.reconstruction_loss_sum += self.reconstruction_loss
                            self.kl_loss_sum += self.kl_loss
                            pbar.update(1)
                            pbar.set_postfix({'Generator Loss': float(self.generator_loss_batch), 'Reconstruction Loss': float(self.reconstruction_loss), 'KL Loss': float(self.kl_loss)})

                # Train LSTM every `lstm_interval` epochs after `vae_epochs`
                if self.epoch >= self.vae_epochs and (self.epoch - self.vae_epochs) % self.lstm_interval == 0:
                    for self.data in self.processed_data:
                        self.augmented_data = self.vae.decode(scbetavaegan.tf.random.normal(shape=(self.data.shape[0], self.latent_dim))).numpy()
                        self.real_data = scbetavaegan.tf.expand_dims(self.data, axis=0)
                        self.generated_data = scbetavaegan.tf.expand_dims(self.augmented_data, axis=0)
                        self.lstm_loss = scbetavaegan.train_lstm_step(self.lstm_discriminator, self.real_data, self.generated_data, self.lstm_optimizer)
                    print(f'LSTM training at epoch {self.epoch+1}: Discriminator Loss = {self.lstm_loss.numpy()}')


                self.avg_generator_loss = self.generator_loss / self.num_batches  # Update the average calculation
                self.avg_reconstruction_loss = self.reconstruction_loss_sum / self.num_batches
                self.avg_kl_loss = self.kl_loss_sum / self.num_batches

                self.generator_loss_history.append(self.avg_generator_loss)  # Update history list
                self.reconstruction_loss_history.append(self.avg_reconstruction_loss)
                self.kl_loss_history.append(self.avg_kl_loss)

                # Calculate NRMSE
                self.nrmse_sum = 0
                for self.data in self.processed_data:
                    self.augmented_data = self.vae.decode(scbetavaegan.tf.random.normal(shape=(self.data.shape[0], self.latent_dim))).numpy()
                    self.rmse = np.sqrt(mean_squared_error(self.data[:, :2], self.augmented_data[:, :2]))
                    self.nrmse = self.rmse / (self.data[:, :2].max() - self.data[:, :2].min())
                    self.nrmse_sum += self.nrmse
                
                self.nrmse_avg = self.nrmse_sum / len(self.processed_data)

                self.nrmse_history.append(self.nrmse_avg)

                print(f"Epoch {self.epoch+1}: Generator Loss = {self.avg_generator_loss:.6f}, Reconstruction Loss = {self.avg_reconstruction_loss:.6f}, KL Divergence Loss = {self.avg_kl_loss:.6f}")
                print(f"NRMSE = {self.nrmse_avg:.6f}")



                # Cell 5 (visualization part)
                if (self.epoch + 1) % self.visual_per_num_epoch == 0:
                    self.base_latent_variability = 100.0
                    self.latent_variability_range = (0.1, 5.0)
                    self.num_augmented_files = 3

                    self.augmented_datasets = scbetavaegan.generate_augmented_data(self.data_frames, self.vae, self.num_augmented_files, self.avg_data_points, self.processed_data, 
                                                                self.base_latent_variability, self.latent_variability_range)

                    # Calculate actual latent variabilities and lengths used
                    self.latent_variabilities = [self.base_latent_variability * np.random.uniform(self.latent_variability_range[0], self.latent_variability_range[1]) for _ in range(self.num_augmented_files)]
                    self.augmented_lengths = [len(self.data) for self.data in self.augmented_datasets]

                    self.fig, self.axs = plt.subplots(1, self.num_augmented_files + len(self.original_data_frames), figsize=(6*(self.num_augmented_files + len(self.original_data_frames)), 6))

                    for self.i, self.original_data in enumerate(self.original_data_frames):
                        self.original_on_paper = self.original_data[self.original_data['pen_status'] == 1]
                        self.original_in_air = self.original_data[self.original_data['pen_status'] == 0]

                        self.axs[i].scatter(self.original_on_paper['y'], self.original_on_paper['x'], c='b', s=1, label='On Paper')
                        self.axs[i].scatter(self.original_in_air['y'], self.original_in_air['x'], c='r', s=1, label='In Air')
                        self.axs[i].set_title(f'Original Data {i+1}')
                        self.axs[i].invert_xaxis()

                    # Set consistent axis limits for square aspect ratio for both original and augmented data
                    self.x_min = min(data[:, 0].min() for data in self.processed_data)
                    self.x_max = max(data[:, 0].max() for data in self.processed_data)
                    self.y_min = min(data[:, 1].min() for data in self.processed_data)
                    self.y_max = max(data[:, 1].max() for data in self.processed_data)

                    for i, (self.augmented_data, self.latent_var, self.length) in enumerate(zip(self.augmented_datasets, self.latent_variabilities, self.augmented_lengths)):
                        self.augmented_on_paper = self.augmented_data[self.augmented_data[:, 3] == 1]
                        self.augmented_in_air = self.augmented_data[self.augmented_data[:, 3] == 0]

                        self.axs[i+len(self.original_data_frames)].scatter(self.augmented_on_paper[:, 1], self.augmented_on_paper[:, 0], c='b', s=1, label='On Paper')
                        self.axs[i+len(self.original_data_frames)].scatter(self.augmented_in_air[:, 1], self.augmented_in_air[:, 0], c='r', s=1, label='In Air')
                        self.axs[i+len(self.original_data_frames)].invert_xaxis()
                        self.axs[i+len(self.original_data_frames)].set_xlim(self.y_max, self.y_min)
                        self.axs[i+len(self.original_data_frames)].set_ylim(self.x_min, self.x_max)

                    plt.tight_layout()
                    
                    # Save VAE model after each epoch, directly into the `vae_models` folder
                    self.model_save_path = os.path.join(self.save_dir, f"epoch_{self.epoch+1}_model.h5")
                    self.vae.save(self.model_save_path)
                    print(f"VAE model saved for epoch {self.epoch+1} at {self.model_save_path}.")

            # Final output and plots
            plt.ioff()
            

            self.vae.save('pentab_saved_model.h5')
            print("Final VAE model saved.")

            # Plot generator loss history
            plt.figure(figsize=(10, 5))
            plt.plot(self.generator_loss_history, label='Generator Loss')  # Update label
            plt.plot(self.reconstruction_loss_history, label='Reconstruction Loss')
            plt.plot(self.kl_loss_history, label='KL Divergence Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            

            # Plot NRMSE history
            plt.subplot(1, 3, 3)
            plt.plot(self.nrmse_history, label='NRMSE')
            plt.xlabel('Epoch')
            plt.ylabel('NRMSE')
            plt.title('Normalized Root Mean Squared Error Over Epochs')
            plt.legend()

            plt.tight_layout()
            
            # Start error
            with custom_object_scope({'VAE': scbetavaegan.VAE}):
                self.vae_pretrained = load_model('model/vae_models/epoch_200_model.h5')
            print("Pretrained VAE model loaded.")

            # Base latent variability settings
            self.base_latent_variability = 100.0
            self.latent_variability_range = (0.99, 1.01)
            self.num_augmented_files = 3

            # Generate augmented data using the pretrained model
            self.augmented_datasets = scbetavaegan.generate_augmented_data(self.vae_pretrained, self.num_augmented_files, self.avg_data_points, self.processed_data, 
                                                        self.base_latent_variability, self.latent_variability_range)

            # Calculate actual latent variabilities and lengths used
            self.latent_variabilities = [self.base_latent_variability * np.random.uniform(self.latent_variability_range[0], self.latent_variability_range[1]) for _ in range(self.num_augmented_files)]
            self.augmented_lengths = [len(self.data) for self.data in self.augmented_datasets]

            # Visualize the original and augmented data side by side
            self.fig, self.axs = plt.subplots(1, self.num_augmented_files + len(self.original_data_frames), figsize=(6 * (self.num_augmented_files + len(self.original_data_frames)), 6))

            # Plot the original data before imputation, with a 90-degree left rotation and horizontal flip
            for self.i, self.original_data in enumerate(self.original_data_frames):  # Use original_data_frames for raw data visualization
                self.original_on_paper = self.original_data[self.original_data['pen_status'] == 1]
                self.original_in_air = self.original_data[self.original_data['pen_status'] == 0]
                
                # Scatter plot for the original data (before imputation), with rotated axes
                self.axs[i].scatter(self.original_on_paper['y'], self.original_on_paper['x'], c='b', s=1, label='On Paper')  # y -> x, x -> y
                self.axs[i].scatter(self.original_in_air['y'], self.original_in_air['x'], c='r', s=1, label='In Air')  # y -> x, x -> y
                self.axs[i].set_title(f'Original Data {i + 1}')
                self.axs[i].set_xlabel('y')  # Previously 'x'
                self.axs[i].set_ylabel('x')  # Previously 'y'
                self.axs[i].set_aspect('equal')
                self.axs[i].legend()
                
                # Flip the horizontal axis (y-axis)
                self.axs[i].invert_xaxis()  # This reverses the 'y' axis to flip the plot horizontally

            # Set consistent axis limits for square aspect ratio for both original and augmented data
            self.x_min = min(self.data['x'].min() for self.data in self.original_data_frames)
            self.x_max = max(self.data['x'].max() for self.data in self.original_data_frames)
            self.y_min = min(self.data['y'].min() for self.data in self.original_data_frames)
            self.y_max = max(self.data['y'].max() for self.data in self.original_data_frames)

            # Plot the augmented data with the same 90-degree left rotation and horizontal flip
            self.all_augmented_data = scbetavaegan.visualize_augmented_data(self.augmented_datasets, self.scalers, self.original_data_frames, self.axs)

            plt.tight_layout()
            
            # End error
                
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())

class Workplace(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Workplace, self).__init__(parent)
        self.uploaded_files = []
        self.setupUi()
        self.worker = None
        

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setAlignment(QtCore.Qt.AlignTop)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.setFont(font)

        # Create a scroll area
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # Create a container widget for the scroll area content
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)

        # Set a size policy for the scroll widget that allows it to shrink
        self.scroll_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        # Add the scroll area to the main layout
        self.scroll_area.setWidget(self.scroll_widget)
        self.gridLayout.addWidget(self.scroll_area)

        # Call functions to set up collapsible components
        self.setup_input_collapsible()
        self.setup_model_collapsible()
        self.setup_preview_collapsible()
        self.setup_process_log_collapsible()
        self.setup_output_collapsible()
        self.setup_result_collapsible()

        # Generate Synthetic Data button
        button_layout = QtWidgets.QVBoxLayout()
        self.generate_data_button = QtWidgets.QPushButton("Generate Synthetic Data", self)
        self.generate_data_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 14px; 
                font-weight: 600; 
                padding: 10px 20px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005555; 
            }
            """
        )
        self.generate_data_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor)) # put the button at the bottom
        self.generate_data_button.clicked.connect(self.on_generate_data)

        button_layout.addWidget(self.generate_data_button, alignment=QtCore.Qt.AlignCenter)

        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        button_layout.addItem(spacer)

        # Adding the button to the main layout
        self.gridLayout.addLayout(button_layout, 1, 0)
        
    def on_generate_data(self):
        # Disable the generate button and change text
        self.generate_data_button.setEnabled(False)
        self.generate_data_button.setText("Generating...")
        
        # Create and start the worker thread
        self.worker = GenerateDataWorker(self)
        
        # Connect signals
        self.worker.finished.connect(self.on_generation_complete)
        self.worker.error.connect(self.on_generation_error)
        self.worker.progress.connect(self.logger.info)  # Connect directly to logger.info
        
        # Start the thread
        self.worker.start()
    
    def on_generation_complete(self):
        # Re-enable the generate button
        self.generate_data_button.setEnabled(True)
        self.generate_data_button.setText("Generate Synthetic Data")
        
        # Clean up
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        # Expand relevant sections
        QtCore.QTimer.singleShot(0, lambda: self.collapsible_widget_process_log.toggle_container(True))
        QtCore.QTimer.singleShot(3000, lambda: self.collapsible_widget_output.toggle_container(True))
        QtCore.QTimer.singleShot(4000, lambda: self.collapsible_widget_result.toggle_container(True))
        
    def on_generation_error(self, error_message):
        # Re-enable the generate button
        self.generate_data_button.setEnabled(True)
        self.generate_data_button.setText("Generate Synthetic Data")
        
        # Show error message
        self.logger.error(f"Error during generation: {error_message}")
        
        # Clean up
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        QMessageBox.critical(
            self,
            'Generation Error',
            f"An error occurred during data generation:\n{error_message}",
            QMessageBox.Ok
        )

    def setup_input_collapsible(self):
        """Set up the 'Input' collapsible widget and its contents."""
        font = QtGui.QFont()
        font.setPointSize(20)

        # Call the collapsible widget component for Input
        self.collapsible_widget_input = CollapsibleWidget("Input", self)
        self.scroll_layout.addWidget(self.collapsible_widget_input)

        # Add the FileUploadWidget
        self.file_upload_widget = DragDrop_Button(self)
        self.file_upload_widget.file_uploaded.connect(self.update_file_display)  # Connect the signal
        self.collapsible_widget_input.add_widget(self.file_upload_widget)

        # Add "Add More Files" button to Input collapsible widget
        self.add_file_button = QtWidgets.QPushButton("Add More Files", self)
        self.add_file_button.setStyleSheet(
            """
            QPushButton {
                background-color: #003333; 
                color: white; 
                font-family: Montserrat; 
                font-size: 14px; 
                font-weight: 600; 
                padding: 8px 16px;
                margin-left: 15px; 
                margin-right: 15px; 
                border-radius: 5px; 
                border: none;
            }
            QPushButton:hover {
                background-color: #005555;  /* Change this to your desired hover color */
            }
            """
        )
        self.add_file_button.setFont(font)
        self.add_file_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.add_file_button.clicked.connect(self.add_more_files)
        self.collapsible_widget_input.add_widget(self.add_file_button)
        
        # Create a scrollable area to hold the file widgets
        self.file_scroll_area = QtWidgets.QScrollArea(self)
        self.file_scroll_area.setWidgetResizable(True)
        self.file_scroll_area.setMinimumHeight(150)

        # Create a container to hold the file widgets and its layout
        self.file_container_widget = QtWidgets.QWidget(self)
        self.file_container_layout = QtWidgets.QVBoxLayout(self.file_container_widget)
        self.file_container_layout.setSpacing(0)
        
        # Add the file container widget to the scroll area
        self.file_scroll_area.setWidget(self.file_container_widget)
        self.collapsible_widget_input.add_widget(self.file_scroll_area)
        

        # Initially hide other components
        self.file_upload_widget.setVisible(True)
        self.show_other_components(False)


        # Open the collapsible widget by default
        self.collapsible_widget_input.toggle_container(True)

    
    def show_other_components(self, show=True):
        """Show or hide other components based on file upload."""
        self.add_file_button.setVisible(show)
        self.file_container_widget.setVisible(show)

    def setup_model_collapsible(self):
        self.collapsible_model_container = CollapsibleWidget("Models", self)
        self.scroll_layout.addWidget(self.collapsible_model_container)

        self.model_widget = ModelWidget(self)
        self.collapsible_model_container.add_widget(self.model_widget)
    
    def setup_preview_collapsible(self):
        self.collapsible_widget_preview = CollapsibleWidget("File Preview", self)
        self.scroll_layout.addWidget(self.collapsible_widget_preview)

        self.file_preview_widget = FilePreviewWidget(self)
        self.collapsible_widget_preview.add_widget(self.file_preview_widget)

    def setup_process_log_collapsible(self):
        self.collapsible_widget_process_log = CollapsibleWidget("Process Log", self)
        self.scroll_layout.addWidget(self.collapsible_widget_process_log)

        self.process_log_widget = ProcessLogWidget(self)
        self.logger = self.process_log_widget.get_logger() 
        self.collapsible_widget_process_log.add_widget(self.process_log_widget)

    def setup_output_collapsible(self):
        self.collapsible_widget_output = CollapsibleWidget("Output", self)
        self.scroll_layout.addWidget(self.collapsible_widget_output)

        self.output_widget = OutputWidget(self)
        self.output_widget.clearUI.connect(self.clear_all_ui)
        self.collapsible_widget_output.add_widget(self.output_widget)

    def setup_result_collapsible(self):
        """Set up the 'Result' collapsible widget and its contents."""

        # Call collapsible widget for Result
        self.collapsible_widget_result = CollapsibleWidget("Result", self)
        self.scroll_layout.addWidget(self.collapsible_widget_result)

        self.svc_preview = SVCpreview(self)
        self.collapsible_widget_result.add_widget(self.svc_preview)
        
    def handle_file_removal(self, file_path, file_name):
        """Handle the file removal logic when a file is removed."""
        if file_path in self.uploaded_files:
            # Remove the file from the uploaded_files list
            self.uploaded_files.remove(file_path)  
            print(f"Removed file: {file_name}, remaining files: {self.uploaded_files}")  # Debug statement

            # Update the UI to reflect the removal
            for i in reversed(range(self.file_container_layout.count())):
                widget = self.file_container_layout.itemAt(i).widget()
                if isinstance(widget, FileContainerWidget) and widget.file_name == file_name:
                    widget.remove_file_signal.disconnect()  # Disconnect signal to avoid errors
                    self.file_container_layout.removeWidget(widget)  # Remove the widget from layout
                    widget.deleteLater()  # Schedule the widget for deletion
                    widget.setParent(None)  # Detach widget from its parent
                    break  # Exit after removing the specific file container

            # If no more files, show the file upload widget again
            if not self.uploaded_files:
                self.show_other_components(False)
                self.file_upload_widget.setVisible(True)
                self.file_preview_widget.clear()

            # Update the file container layout to reflect the changes
            self.file_container_layout.update()

    def update_file_display(self, new_uploaded_files):
        """Update the display of files based on newly uploaded files."""
        # Append new files to the existing list, avoiding duplicates
        for file_path in new_uploaded_files:
            if file_path not in self.uploaded_files:
                self.uploaded_files.append(file_path)
        
        print("Uploaded files:", self.uploaded_files)  # Debugging output
        
        has_files = bool(self.uploaded_files)
        self.show_other_components(has_files)

        # Hide the file upload widget if files are uploaded
        self.file_upload_widget.setVisible(not has_files)

        # Clear existing widgets in the file container layout
        for i in reversed(range(self.file_container_layout.count())):
            widget = self.file_container_layout.itemAt(i).widget()
            if widget is not None:
                widget.remove_file_signal.disconnect()  # Disconnect signal to avoid errors
                widget.deleteLater()  # Schedule widget deletion
                self.file_container_layout.removeWidget(widget)

        # Re-add file containers for each uploaded file and update preview
        for file_path in self.uploaded_files:
            file_name = os.path.basename(file_path)

            # Verify the file still exists before displaying it
            if os.path.exists(file_path):
                new_file_container = FileContainerWidget(file_path, self)
                new_file_container.hide_download_button()
                new_file_container.hide_retry_button()
                new_file_container.remove_file_signal.connect(self.handle_file_removal)  # Connect remove signal
                self.file_container_layout.addWidget(new_file_container)

                # Display the file content in the file preview widget
                self.file_preview_widget.display_file_contents(file_path)

                # Display the file content in the result preview widget
                self.svc_preview.display_file_contents(file_path, 0)

        self.file_preview_widget.set_uploaded_files(self.uploaded_files)
        
        # Automatically expand the preview collapsible widget if there are files
        if has_files:
            self.collapsible_widget_preview.toggle_container(True)

    def add_more_files(self):
        self.file_upload_widget.open_file_dialog()
    
    def get_image_path(self, image_name):
        path = os.path.join(os.path.dirname(__file__), '..', 'icon', image_name)
        return path
    
    def clear_all_ui(self):
        # Clear uploaded files
        self.uploaded_files = []
        
        # Reset file upload widget
        self.file_upload_widget.setVisible(True)
        self.show_other_components(False)
        
        # Clear file containers
        for i in reversed(range(self.file_container_layout.count())):
            widget = self.file_container_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Clear previews and logs
        self.file_preview_widget.clear()
        self.process_log_widget.clear()
        self.svc_preview.clear()
        
        # Collapse all widgets except Input
        self.collapsible_widget_preview.toggle_container(False)
        self.collapsible_widget_process_log.toggle_container(False)
        self.collapsible_widget_output.toggle_container(False)
        self.collapsible_widget_result.toggle_container(False)
        
if __name__ == "__main__":
    
    
    app = QtWidgets.QApplication([])
    window = Workplace()
    window.show()
    app.exec_()
