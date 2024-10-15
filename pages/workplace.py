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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope

from PyQt5.QtCore import QThread, pyqtSignal
import traceback

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

from glob import glob
import re

from PyQt5.QtWidgets import QApplication


class GenerateDataWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)  # For logging progress
    generation_complete = pyqtSignal()
    metrics = pyqtSignal(str)

    def __init__(self, workplace):

        super().__init__()
        self.workplace = workplace
        self.uploaded_files = workplace.uploaded_files
        self.model = None
        self.num_augmentations = 1
    
    def set_model(self, model):
        self.model = model

    def set_num_augmentations(self, num_augmentations):
        self.num_augmentations = num_augmentations

    def run(self):
        plt.close('all')
        try:
            self.progress.emit("Starting data generation process...")
            self.progress.emit(f"There are {self.num_augmentations} augmentations to generate per file.")
            # Move all the generation logic here
            self.timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.folder_name = f"SyntheticData_{self.timestamp}"
            self.output_dir = os.path.join(
                os.path.dirname(__file__), "../files/uploads", self.folder_name
            )

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.progress.emit(f"Created output directory: {self.folder_name}")

            self.progress.emit("Copying input files...")
            for file_path in self.uploaded_files:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    destination_path = os.path.join(self.output_dir, file_name)
                    shutil.copy(file_path, destination_path)

            self.progress.emit("Starting synthetic data generation...")

            # Simulate some processing steps with different log levels
            self.progress.emit("Checking input files...")
            time.sleep(1)

            if self.uploaded_files:
                self.progress.emit(f"Found {len(self.uploaded_files)} files to process")

                self.progress.emit("Generating synthetic data...")
                time.sleep(2)
                self.progress.emit("Synthetic data generation completed successfully!")

            else:
                self.progress.emit("Error: No input files found. Please upload files before generating data.")
                QMessageBox.warning(
                    self,
                    "No Files",
                    "No input files uploaded to generate synthetic data",
                    QMessageBox.Ok,
                )
            print(self.uploaded_files)

            self.progress.emit("Starting data preprocessing...")
            self.num_files_to_use = len(self.uploaded_files)
            (
                self.data_frames,
                self.processed_data,
                self.scalers,
                self.avg_data_points,
                self.input_filenames,
                self.original_data_frames
            ) = scbetavaegan.upload_and_process_files(
                self.output_dir, self.num_files_to_use
            )

            self.progress.emit(f"Preprocessed {len(self.processed_data)} files")


            self.original_absolute_files = scbetavaegan.save_original_data(self.original_data_frames, self.input_filenames)

            # # Store the name of the first file for use in Cell 4
            self.input_filename = (
                self.input_filenames[0] if self.input_filenames else "processed_data"
            )
            print(
                f"Number of processed files: {len(self.processed_data)}"
            )  ##dito sa processed data naka_store
            print(f"Average number of data points: {self.avg_data_points}")

            self.progress.emit("Processing time series data and masking gaps...")
            for self.df_idx in range(len(self.data_frames)):
                self.progress.emit(f"Processing file {self.df_idx + 1}/{len(self.data_frames)}")
                self.df = self.data_frames[
                    self.df_idx
                ]  # Using each DataFrame in the list

                # Convert the 'timestamp' column to numeric for calculations (if not already done)
                self.df["timestamp"] = pd.to_numeric(self.df["timestamp"])

                # Sort the DataFrame by timestamp (should already be sorted in the function)
                self.df.sort_values("timestamp", inplace=True)

                # Calculate the differences between consecutive timestamps (optional for gap finding)
                self.df["time_diff"] = self.df["timestamp"].diff()

                # Identify the indices where the time difference is greater than 30,000 milliseconds
                self.gap_indices = self.df.index[self.df["time_diff"] > 8].tolist()

                # Create an empty list to hold the new rows
                self.new_rows = []

                # Fill in the gaps with 70 milliseconds intervals
                for self.idx in self.gap_indices:
                    # Check if the next index is valid
                    if self.idx + 1 < len(self.df):
                        # Get the current and next timestamps
                        self.current_timestamp = self.df.at[self.idx, "timestamp"]
                        self.next_timestamp = self.df.at[self.idx + 1, "timestamp"]

                        # Calculate how many entries we need to fill in
                        self.num_fill_entries = (
                            self.next_timestamp - self.current_timestamp
                        ) // 7

                        # Generate the timestamps to fill the gap
                        for self.i in range(1, self.num_fill_entries + 1):
                            self.new_timestamp = self.current_timestamp + self.i * 7

                            # Create a new row to fill in with NaN for x and y
                            self.new_row = {
                                "x": np.nan,  # Set x to NaN
                                "y": np.nan,  # Set y to NaN
                                "timestamp": self.new_timestamp,
                                "pen_status": 0,  # You can set this to your desired value
                                "azimuth": self.df.at[
                                    self.idx, "azimuth"
                                ],  # Use the current azimuth value
                                "altitude": self.df.at[
                                    self.idx, "altitude"
                                ],  # Use the current altitude value
                                "pressure": self.df.at[
                                    self.idx, "pressure"
                                ],  # Use the current pressure value
                            }

                            # Append the new row to the list of new rows
                            self.new_rows.append(self.new_row)

                # Create a DataFrame from the new rows
                self.new_rows_df = pd.DataFrame(self.new_rows)

                # Concatenate the original DataFrame with the new rows DataFrame
                self.df = pd.concat([self.df, self.new_rows_df], ignore_index=True)

                # Sort the DataFrame by timestamp to maintain order
                self.df.sort_values("timestamp", inplace=True)

                # Reset index after sorting
                self.df.reset_index(drop=True, inplace=True)

                # Check for NaN entries before interpolation
                if self.df[["x", "y"]].isnull().any().any():
                    self.df[["x", "y"]] = self.df[["x", "y"]].interpolate(
                        method="linear"
                    )

                # Drop the 'time_diff' column after processing
                self.df.drop(columns=["time_diff"], inplace=True)

                # Update the processed data
                self.data_frames[self.df_idx] = self.df

            # Update processed data for all DataFrames
            self.processed_data = [
                np.column_stack(
                    (
                        self.scaler.transform(self.df[["x", "y", "timestamp"]]),
                        self.df["pen_status"].values,
                    )
                )
                for self.df, self.scaler in zip(self.data_frames, self.scalers)
            ]
            self.avg_data_points = int(
                np.mean([self.df.shape[0] for self.df in self.data_frames])
            )

            self.imputed_folder = 'files/imputed'
            os.makedirs(self.imputed_folder, exist_ok=True)

            self.processed_dataframes = []

            for self.input_filename, self.df in zip(self.input_filenames, self.data_frames):
                # Convert all numeric columns to integers
                self.df[['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']] = self.df[['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']].astype(int)

                # Save the processed DataFrame to the 'imputed' folder with the same input filename
                self.save_path = os.path.join(self.imputed_folder, self.input_filename)
                self.df.to_csv(self.save_path, sep=' ', index=False, header=False)  # Save without header and index
                # Append the processed DataFrame to the list
                self.processed_dataframes.append(self.df)

                print(f"Processed DataFrame saved as: {self.input_filename}")

            for self.input_filename, self.df in zip(
                self.input_filenames, self.data_frames
            ):
                # Convert all numeric columns to integers
                self.df[
                    [
                        "x",
                        "y",
                        "timestamp",
                        "pen_status",
                        "pressure",
                        "azimuth",
                        "altitude",
                    ]
                ] = self.df[
                    [
                        "x",
                        "y",
                        "timestamp",
                        "pen_status",
                        "pressure",
                        "azimuth",
                        "altitude",
                    ]
                ].astype(
                    int
                )

                # Append the processed DataFrame to the list
                self.processed_dataframes.append(self.df)

                print(f"Processed DataFrame for: {self.input_filename}")

            # Use the processed_dataframes directly
            (
                self.data_frames,
                self.processed_data,
                self.scalers,
                self.avg_data_points,
                self.original_filenames,
            ) = scbetavaegan.process_dataframes(
                self.processed_dataframes, self.num_files_to_use
            )
            print(f"Number of processed files: {len(self.processed_data)}")
            print(f"Average number of data points: {self.avg_data_points}")

            self.progress.emit("Completed gap masking process...")
            self.progress.emit("Initializing VAE and LSTM models...")

            self.latent_dim = 128
            self.beta = 0.0001
            self.learning_rate = 0.001
            self.lambda_shift = 0.5

            self.vae = scbetavaegan.VAE(self.latent_dim, self.beta, self.lambda_shift)
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

            if self.model is None:
                self.train_model()
            else:
                self.generate_synthetic_data(self.model, self.data_frames, self.processed_data, self.scalers, self.avg_data_points, self.input_filenames, self.original_data_frames)

        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())



    def train_model(self):
        try:
            tf.keras.backend.clear_session()
            # Initialize LSTM discriminator and optimizer
            self.lstm_discriminator = scbetavaegan.LSTMDiscriminator()
            self.lstm_optimizer = tf.keras.optimizers.Adam(
                self.learning_rate
            )

            self.batch_size = 512
            self.train_datasets = [
                tf.data.Dataset.from_tensor_slices(self.data)
                .shuffle(10000)
                .batch(self.batch_size)
                for self.data in self.processed_data
            ]

            # Set up alternating epochs
            self.vae_epochs = 200
            self.lstm_interval = 50
            self.epochs = 5
            self.visual_per_num_epoch = 5
            self.num_augmented_files = len(self.uploaded_files)

            self.generator_loss_history = []
            self.reconstruction_loss_history = []
            self.kl_loss_history = []
            self.nrmse_history = []

            self.save_dir = "pre-trained"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            self.progress.emit("Starting model training...")

            for self.epoch in range(self.epochs):
                self.generator_loss = 0
                self.reconstruction_loss_sum = 0
                self.kl_loss_sum = 0
                self.num_batches = sum(
                    len(self.dataset) for self.dataset in self.train_datasets
                )

                with scbetavaegan.tqdm(
                    total=self.num_batches,
                    desc=f"Epoch {self.epoch+1}/{self.epochs}",
                    unit="batch",
                ) as pbar:
                    for dataset in self.train_datasets:
                        for self.batch in dataset:
                            self.use_lstm = (
                                self.epoch >= self.vae_epochs
                                and (self.epoch - self.vae_epochs) % self.lstm_interval
                                == 0
                            )
                            (
                                self.generator_loss_batch,
                                self.reconstruction_loss,
                                self.kl_loss,
                            ) = scbetavaegan.train_vae_step(
                                self.vae,
                                self.batch,
                                self.optimizer,
                                self.lstm_discriminator if self.use_lstm else None,
                            )
                            self.generator_loss += self.generator_loss_batch
                            self.reconstruction_loss_sum += self.reconstruction_loss
                            self.kl_loss_sum += self.kl_loss
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "Generator Loss": float(self.generator_loss_batch),
                                    "Reconstruction Loss": float(
                                        self.reconstruction_loss
                                    ),
                                    "KL Loss": float(self.kl_loss),
                                }
                            )

                # Train LSTM every `lstm_interval` epochs after `vae_epochs`
                if (
                    self.epoch >= self.vae_epochs
                    and (self.epoch - self.vae_epochs) % self.lstm_interval == 0
                ):
                    for self.data in self.processed_data:
                        self.augmented_data = self.vae.decode(
                            tf.random.normal(
                                shape=(self.data.shape[0], self.latent_dim)
                            )
                        ).numpy()
                        self.real_data = tf.expand_dims(self.data, axis=0)
                        self.generated_data = tf.expand_dims(
                            self.augmented_data, axis=0
                        )
                        self.lstm_loss = scbetavaegan.train_lstm_step(
                            self.lstm_discriminator,
                            self.real_data,
                            self.generated_data,
                            self.lstm_optimizer,
                        )
                    self.progress.emit(
                        f"LSTM training at epoch {self.epoch+1}: Discriminator Loss = {self.lstm_loss.numpy()}"
                    )
                    print(
                        f"LSTM training at epoch {self.epoch+1}: Discriminator Loss = {self.lstm_loss.numpy()}"
                    )

                self.avg_generator_loss = (
                    self.generator_loss / self.num_batches
                )  # Update the average calculation
                self.avg_reconstruction_loss = (
                    self.reconstruction_loss_sum / self.num_batches
                )
                self.avg_kl_loss = self.kl_loss_sum / self.num_batches

                self.generator_loss_history.append(
                    self.avg_generator_loss
                )  # Update history list
                self.reconstruction_loss_history.append(self.avg_reconstruction_loss)
                self.kl_loss_history.append(self.avg_kl_loss)

                print(
                    f"Epoch {self.epoch+1}: Generator Loss = {self.avg_generator_loss:.6f}, Reconstruction Loss = {self.avg_reconstruction_loss:.6f}, KL Divergence Loss = {self.avg_kl_loss:.6f}"
                )
                self.progress.emit(f"Training Epoch {self.epoch+1}: Generator Loss = {self.avg_generator_loss:.6f}, Reconstruction Loss = {self.avg_reconstruction_loss:.6f}, KL Divergence Loss = {self.avg_kl_loss:.6f}")

            
                # Cell 5 (visualization part)
                if (self.epoch + 1) % self.visual_per_num_epoch == 0:
                    self.base_latent_variability = 100.0
                    self.latent_variability_range = (0.1, 5.0)
                    self.num_augmented_files = len(self.uploaded_files)

                    self.augmented_datasets = scbetavaegan.generate_augmented_data(
                        self.data_frames,
                        self.vae,
                        self.num_augmented_files,
                        self.avg_data_points,
                        self.processed_data,
                        self.base_latent_variability,
                        self.latent_variability_range,
                    )

                    # Calculate actual latent variabilities and lengths used
                    self.latent_variabilities = [
                        self.base_latent_variability
                        * np.random.uniform(
                            self.latent_variability_range[0],
                            self.latent_variability_range[1],
                        )
                        for _ in range(self.num_augmented_files)
                    ]
                    self.augmented_lengths = [
                        len(self.data) for self.data in self.augmented_datasets
                    ]

                    self.fig, self.axs = plt.subplots(
                        1,
                        self.num_augmented_files + len(self.original_data_frames),
                        figsize=(
                            6
                            * (
                                self.num_augmented_files
                                + len(self.original_data_frames)
                            ),
                            6,
                        ),
                    )

                    for self.i, self.original_data in enumerate(
                        self.original_data_frames
                    ):
                        self.original_on_paper = self.original_data[
                            self.original_data["pen_status"] == 1
                        ]
                        self.original_in_air = self.original_data[
                            self.original_data["pen_status"] == 0
                        ]

                        self.axs[self.i].scatter(
                            self.original_on_paper["y"],
                            self.original_on_paper["x"],
                            c="b",
                            s=1,
                            label="On Paper",
                        )
                        self.axs[self.i].scatter(
                            self.original_in_air["y"],
                            self.original_in_air["x"],
                            c="r",
                            s=1,
                            label="In Air",
                        )
                        self.axs[self.i].set_title(f"Original Data {self.i+1}")
                        self.axs[self.i].invert_xaxis()

                    # Set consistent axis limits for square aspect ratio for both original and augmented data
                    self.x_min = min(self.data[:, 0].min() for self.data in self.processed_data)
                    self.x_max = max(self.data[:, 0].max() for self.data in self.processed_data)
                    self.y_min = min(self.data[:, 1].min() for self.data in self.processed_data)
                    self.y_max = max(self.data[:, 1].max() for self.data in self.processed_data)

                    for self.i, (
                        self.augmented_data,
                        self.latent_var,
                        self.length,
                    ) in enumerate(
                        zip(
                            self.augmented_datasets,
                            self.latent_variabilities,
                            self.augmented_lengths,
                        )
                    ):
                        self.augmented_on_paper = self.augmented_data[
                            self.augmented_data[:, 3] == 1
                        ]
                        self.augmented_in_air = self.augmented_data[
                            self.augmented_data[:, 3] == 0
                        ]

                        self.axs[self.i + len(self.original_data_frames)].scatter(
                            self.augmented_on_paper[:, 1],
                            self.augmented_on_paper[:, 0],
                            c="b",
                            s=1,
                            label="On Paper",
                        )
                        self.axs[self.i + len(self.original_data_frames)].scatter(
                            self.augmented_in_air[:, 1],
                            self.augmented_in_air[:, 0],
                            c="r",
                            s=1,
                            label="In Air",
                        )
                        self.axs[self.i + len(self.original_data_frames)].invert_xaxis()
                        self.axs[self.i + len(self.original_data_frames)].set_xlim(
                            self.y_max, self.y_min
                        )
                        self.axs[self.i + len(self.original_data_frames)].set_ylim(
                            self.x_min, self.x_max
                        )

                    plt.tight_layout()

            # Plot generator loss history
            plt.figure(figsize=(10, 5))
            plt.plot(
                self.generator_loss_history, label="Generator Loss"
            )  # Update label
            plt.plot(self.reconstruction_loss_history, label="Reconstruction Loss")
            plt.plot(self.kl_loss_history, label="KL Divergence Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Epochs")
            plt.legend()

            plt.tight_layout()

            self.progress.emit("Model training completed. Saving model...")

            # Get the number of files in the 'pre-trained' folder
            self.num_files_pretrained = len([name for name in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, name))])

            # Save VAE model after each epoch, directly into the `vae_models` folder
            self.model_save_path = os.path.join(self.save_dir, f"Pretrained_Model_{self.num_files_pretrained+1}.h5")
            self.vae.save(self.model_save_path)
            print(f"VAE model saved at {self.model_save_path}.")

            self.progress.emit("Starting synthetic data generation...")
            self.generate_synthetic_data(os.path.basename(self.model_save_path), self.data_frames, self.processed_data, self.scalers, self.avg_data_points, self.input_filenames, self.original_data_frames)

        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())
        
    def generate_synthetic_data(self, pretrained_filename, data_frames, processed_data, scalers, avg_data_points, input_filenames, original_data_frames):
        try:
            # Base latent variability settings
            self.base_latent_variability = 100.0
            self.latent_variability_range = (0.99, 1.01)
            self.all_augmented_data = [] 

            self.progress.emit(f"Generating synthetic data using {pretrained_filename}...")
            self.all_augmented_filepaths, self.augmented_zip_filepath = scbetavaegan.nested_augmentation(self.all_augmented_data, self.num_augmentations, self.num_files_to_use, pretrained_filename, self.base_latent_variability, self.latent_variability_range, data_frames, processed_data, scalers, avg_data_points, input_filenames, original_data_frames)
            
            self.progress.emit("Synthetic data generation completed.")
            self.generation_complete.emit()
            self.result_preview(input_filenames)
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())

    def result_preview(self, input_filenames):
        try:
            self.progress.emit("Starting result preview and analysis...")
            # Define the folders directly in the notebook cell
            self.imputed_folder = "files/imputed"
            self.augmented_folder = "files/augmented_data"

            # Process the files and calculate NRMSE
            self.progress.emit("Calculating NRMSE for generated data...")
            self.results = scbetavaegan.process_files_NRMSE(self.imputed_folder, self.augmented_folder, input_filenames)
            # Display the results
            self.progress.emit("=== RESULTS: NRMSE ANALYSIS ===")
            for self.original_file, self.nrmse_values in self.results.items():
                print(f"Results for {self.original_file}:")
                self.progress.emit(f"Results for {self.original_file}:")
                QApplication.processEvents()
                for i, self.nrmse in enumerate(self.nrmse_values):
                    self.augmented_version = f"({i})" if i > 0 else "base"
                    QApplication.processEvents()
                    print(f"  NRMSE for augmented version {self.augmented_version}: {self.nrmse:.4f}")
                
                if self.nrmse_values:
                    self.avg_nrmse = np.mean(self.nrmse_values)
                    self.progress.emit(f"  Average NRMSE: {self.avg_nrmse:.4f}")
                    QApplication.processEvents()
                    print(f"  Average NRMSE: {self.avg_nrmse:.4f}")
                print()
                self.progress.emit("")

            # Calculate and display the overall average NRMSE
            self.all_nrmse = [self.nrmse for self.nrmse_list in self.results.values() for self.nrmse in self.nrmse_list]
            self.overall_avg_nrmse = np.mean(self.all_nrmse)
            print(f"Overall Average NRMSE: {self.overall_avg_nrmse:.4f}")
            self.progress.emit(f"Overall Average NRMSE: {self.overall_avg_nrmse:.4f}")
            self.metrics.emit("NRMSE")
            self.progress.emit("=== END OF NRMSE ANALYSIS ===")

            self.progress.emit("Calculating post-hoc discriminative score...")
            # Process files, without NRMSE
            self.real_data, self.synthetic_data = scbetavaegan.process_files_PHDS(self.imputed_folder, self.augmented_folder, input_filenames)

            # Compute post-hoc discriminative score
            self.mean_accuracy, self.std_accuracy = scbetavaegan.post_hoc_discriminative_score(self.real_data, self.synthetic_data)

            print(f"Mean accuracy: {self.mean_accuracy:.4f} (±{self.std_accuracy:.4f})")
            self.progress.emit("=== RESULTS: POST-HOC DISCRIMINATIVE SCORE ===")
            self.progress.emit(f"Mean accuracy: {self.mean_accuracy:.4f} (±{self.std_accuracy:.4f})")
            self.metrics.emit("PHDS")
            self.progress.emit("=== END OF POST-HOC DISCRIMINATIVE SCORE ===")

            self.progress.emit("Calculating post-hoc predictive score...")

            self.X, self.y, self.scaler = scbetavaegan.prepare_data(self.data_frames[0])

            self.kf = KFold(n_splits=10, shuffle=True, random_state=np.random.randint(1000))  # 10-fold cross-validation

            self.mape_values = []
            for self.fold, (self.train_index, self.test_index) in enumerate(self.kf.split(self.X), start=1):
                print(f"\n--- Fold {self.fold} ---")
                self.progress.emit(f"Processing fold {self.fold} of 10...")
                QApplication.processEvents()
                
                # Split data into training and testing sets for this fold
                self.X_train, self.X_test = self.X[self.train_index], self.X[self.test_index]
                self.y_train, self.y_test = self.y[self.train_index], self.y[self.test_index]

                self.model = scbetavaegan.create_model((self.X_train.shape[1], self.X_train.shape[2]))
                self.model.fit(self.X_train, self.y_train, epochs=2, batch_size=1024, verbose=3, callbacks=[scbetavaegan.CustomCallback()])
                
                # Evaluate the model and store MAPE
                self.mape = scbetavaegan.evaluate_model(self.model, self.X_test, self.y_test, self.scaler)
                print(f"Fold {self.fold} MAPE: {self.mape * 100:.2f}%")  # Print MAPE for the current fold
                self.mape_values.append(self.mape)

            # Step 5: Calculate Mean and Standard Deviation of MAPE
            self.mean_mape = np.mean(self.mape_values)
            self.std_mape = np.std(self.mape_values)

            self.progress.emit("=== RESULTS: POST-HOC PREDICTIVE SCORE ===")
            self.progress.emit(f"Mean MAPE: {self.mean_mape * 100:.2f}%")
            self.progress.emit(f"Standard Deviation of MAPE: {self.std_mape * 100:.2f}%")
            self.metrics.emit("PHPS")
            self.progress.emit("=== END OF POST-HOC PREDICTIVE SCORE ===")

            self.progress.emit("Result preview and analysis completed.")

            print(f"\nMean MAPE: {self.mean_mape * 100:.2f}%")
            print(f"Standard Deviation of MAPE: {self.std_mape * 100:.2f}%")

            self.finished.emit()
                        
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


class Workplace(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Workplace, self).__init__(parent)
        self.uploaded_files = []
        self.setupUi()
        self.worker = None
        self.has_files = False

    def setupUi(self):
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setAlignment(QtCore.Qt.AlignTop)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.setFont(font)

        # Create a scroll area
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        # Create a container widget for the scroll area content
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)

        # Set a size policy for the scroll widget that allows it to shrink
        self.scroll_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )

        # Add the scroll area to the main layout
        self.scroll_area.setWidget(self.scroll_widget)
        self.gridLayout.addWidget(self.scroll_area)

        # Call functions to set up collapsible components
        self.setup_input_collapsible()
        self.setup_preview_collapsible()
        self.setup_model_collapsible()
        self.setup_process_log_collapsible()
        self.setup_output_collapsible()
        self.setup_result_collapsible()

        # Generate Synthetic Data button
        button_layout = QtWidgets.QVBoxLayout()
        self.generate_data_button = QtWidgets.QPushButton(
            "Generate Synthetic Data", self
        )
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
        self.generate_data_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )  # put the button at the bottom
        self.generate_data_button.clicked.connect(self.on_generate_data)

        button_layout.addWidget(
            self.generate_data_button, alignment=QtCore.Qt.AlignCenter
        )

        spacer = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        button_layout.addItem(spacer)

        # Adding the button to the main layout
        self.gridLayout.addLayout(button_layout, 1, 0)

    def train_vae(self):
        confirmation = self.model_widget.create_custom_message_box(
            title="Train SC-β-VAE-GAN",
            message=f"Are you sure you want to train a new model?"
        )
        
        # Proceed only if the user confirms with 'Yes'
        if confirmation:
            self.process_log_widget.clear()
            self.model_widget.uncheck_checkbox()
            self.svc_preview.clear()
            self.collapsible_widget_output.toggle_container(False)
            self.collapsible_widget_result.toggle_container(False)
            self.collapsible_widget_process_log.toggle_container(True)

            # Disable the generate button and change text
            self.generate_data_button.setEnabled(False)
            self.generate_data_button.setText("Generating...")

            # Create and start the worker thread
            self.worker = GenerateDataWorker(self)
            self.worker.set_model(None)
            self.num_augmentations = self.model_widget.slider_widget.getValue()
            self.worker.set_num_augmentations(self.num_augmentations)

            # Connect signals
            self.worker.generation_complete.connect(self.on_generation_finished)
            self.worker.finished.connect(self.on_generation_complete)
            self.worker.error.connect(self.on_generation_error)
            self.worker.progress.connect(
                self.logger.info
            )  # Connect directly to logger.info
            self.worker.metrics.connect(self.on_generation_results)

            # Start the thread
            self.worker.start()


    def on_generate_data(self):
        self.selected_model = self.model_widget.current_checked_file

        if self.has_files is False:
            self.show_error("Please upload a file first")
        elif self.selected_model == None:
            self.show_error("Please select a pre-trained model first or train your own model")
        elif self.has_files is True and self.selected_model != None:
            self.process_log_widget.clear()
            self.svc_preview.clear()
            self.collapsible_widget_output.toggle_container(False)
            self.collapsible_widget_result.toggle_container(False)
            self.collapsible_widget_process_log.toggle_container(True)
            if self.selected_model == "EMOTHAW.h5":
                self.svc_preview.add_graph_containers()
            # Disable the generate button and change text
            self.generate_data_button.setEnabled(False)
            self.generate_data_button.setText("Generating...")

            # Create and start the worker thread
            self.worker = GenerateDataWorker(self)
            self.worker.set_model(self.selected_model)
            self.num_augmentations = self.model_widget.slider_widget.getValue()
            self.worker.set_num_augmentations(self.num_augmentations)

            # Connect signals
            self.worker.error.connect(self.on_generation_error)
            self.worker.progress.connect(self.logger.info)  # Connect directly to logger.info
            self.worker.generation_complete.connect(self.on_generation_finished)
            self.worker.finished.connect(self.on_generation_complete)
            self.worker.metrics.connect(self.on_generation_results)

            # Start the worker
            self.worker.start()

    def on_generation_complete(self):
        # Re-enable the generate button
        self.generate_data_button.setEnabled(True)
        self.generate_data_button.setText("Generate Synthetic Data")

        self.model_widget.refresh_file_list()

        # Clean up
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    
    def on_generation_finished(self):
        # Disable the generate button and change text
        self.generate_data_button.setEnabled(False)
        self.generate_data_button.setText("Calculating Results...")

        self.update_output_file_display(self.worker.all_augmented_filepaths)
        self.update_original_absolute_file_display(self.worker.original_absolute_files)
        self.output_widget.set_zip_path(self.worker.augmented_zip_filepath)

        # Expand relevant sections
        self.collapsible_widget_output.toggle_container(True)
        self.collapsible_widget_result.toggle_container(True)

    def on_generation_results(self, results):
        if results == "NRMSE":
            self.svc_preview.add_result_text("Normalized Root Mean Square Error (NRMSE)")
            self.svc_preview.add_result_text(f"\tOverall Average NRMSE: {self.worker.overall_avg_nrmse:.4f}")
        elif results == "PHDS":
            self.svc_preview.add_result_text("\nPost-Hoc Discriminative Score (PHDS)")
            self.svc_preview.add_result_text(f"\tMean accuracy: {self.worker.mean_accuracy:.4f} (±{self.worker.std_accuracy:.4f})")
        elif results == "PHPS":
            self.svc_preview.add_result_text("\nPost-Hoc Predictive Score (PHPS)")
            self.svc_preview.add_result_text(f"\tMean MAPE: {self.worker.mean_mape * 100:.2f}%")
            self.svc_preview.add_result_text(f"\tStandard Deviation of MAPE: {self.worker.std_mape * 100:.2f}%")

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
            "Generation Error",
            f"An error occurred during data generation:\n{error_message}",
            QMessageBox.Ok,
        )

    def show_error(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(message)

        if message == "Please upload a file first":
            msg.setWindowTitle("File Upload Error")
        else:
            msg.setWindowTitle("Model Selection Error")
        
        # Set custom icon
        icon = QIcon("icon/icon.ico")
        msg.setWindowIcon(icon)

        msg.exec_()

    def setup_input_collapsible(self):
        """Set up the 'Input' collapsible widget and its contents."""
        font = QtGui.QFont()
        font.setPointSize(20)

        # Call the collapsible widget component for Input
        self.collapsible_widget_input = CollapsibleWidget("Input", self)
        self.scroll_layout.addWidget(self.collapsible_widget_input)

        # Add the FileUploadWidget
        self.file_upload_widget = DragDrop_Button(self)
        self.file_upload_widget.file_uploaded.connect(
            self.update_file_display
        )  # Connect the signal
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
        self.file_scroll_area.setMinimumHeight(0)

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
        self.model_widget.train_button.clicked.connect(self.train_vae)
        self.selected_model = self.model_widget.current_checked_file
        self.num_augmentations = self.model_widget.slider_widget.getValue()
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
        # Add the Output Widget
        self.collapsible_widget_output = CollapsibleWidget("Output", self)
        self.scroll_layout.addWidget(self.collapsible_widget_output)
        self.output_widget = OutputWidget(self)
        self.collapsible_widget_output.add_widget(self.output_widget)

    def setup_result_collapsible(self):
        """Set up the 'Result' collapsible widget and its contents."""

        # Call collapsible widget for Result
        self.collapsible_widget_result = CollapsibleWidget("Result", self)
        self.scroll_layout.addWidget(self.collapsible_widget_result)

        self.svc_preview = SVCpreview(self)
        self.collapsible_widget_result.add_widget(self.svc_preview)
    
    def handle_checkbox_click(self, filename, state):
        if state == QtCore.Qt.Checked:
            self.selected_filename = filename
            print(f"Selected file: {self.selected_filename}")
        else:
            if self.selected_filename == filename:
                self.selected_filename = None
            print(f"Deselected file: {filename}")

    def handle_file_removal(self, file_path, file_name):
        """Handle the file removal logic when a file is removed."""
        if file_path in self.uploaded_files:
            # Remove the file from the uploaded_files list
            self.uploaded_files.remove(file_path)
            print(
                f"Removed file: {file_name}, remaining files: {self.uploaded_files}"
            )  # Debug statement

            # Update the UI to reflect the removal
            for i in reversed(range(self.file_container_layout.count())):
                widget = self.file_container_layout.itemAt(i).widget()
                if (
                    isinstance(widget, FileContainerWidget)
                    and widget.file_name == file_name
                ):
                    widget.remove_file_signal.disconnect()  # Disconnect signal to avoid errors
                    self.file_container_layout.removeWidget(
                        widget
                    )  # Remove the widget from layout
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
            self.has_files = bool(self.uploaded_files)
            if self.has_files == False:
                self.clear_all_ui()

    def update_file_display(self, new_uploaded_files):
        """Update the display of files based on newly uploaded files."""
        # Append new files to the existing list, avoiding duplicates
        for file_path in new_uploaded_files:
            if file_path not in self.uploaded_files:
                self.uploaded_files.append(file_path)

        print("Uploaded files:", self.uploaded_files)  # Debugging output

        self.has_files = bool(self.uploaded_files)
        self.show_other_components(self.has_files)

        # Hide the file upload widget if files are uploaded
        self.file_upload_widget.setVisible(not self.has_files)

        # Clear existing widgets in the file container layout
        for i in reversed(range(self.file_container_layout.count())):
            widget = self.file_container_layout.itemAt(i).widget()
            if widget is not None:
                widget.remove_file_signal.disconnect()  # Disconnect signal to avoid errors
                widget.deleteLater()  # Schedule widget deletion
                self.file_container_layout.removeWidget(widget)

        # Re-add file containers for each uploaded file and update preview
        for index, file_path in enumerate(self.uploaded_files):
            file_name = os.path.basename(file_path)

            # Verify the file still exists before displaying it
            if os.path.exists(file_path):
                new_file_container = FileContainerWidget(file_path, self)
                new_file_container.hide_download_button()
                new_file_container.hide_retry_button()
                new_file_container.remove_file_signal.connect(self.handle_file_removal)  # Connect remove signal
                self.file_container_layout.addWidget(new_file_container)

                # Check if this is the first file
                if index == 0:  # This means it's the first file
                    # Display the file content in the file preview widget
                    self.file_preview_widget.display_file_contents(file_path)

        self.file_preview_widget.set_uploaded_files(self.uploaded_files)
        self.svc_preview.set_uploaded_files(self.uploaded_files)

        # Automatically expand the preview collapsible widget if there are files
        if self.has_files:
            self.collapsible_model_container.toggle_container(True)
            self.collapsible_widget_preview.toggle_container(True)


    def update_output_file_display(self, all_augmented_filepaths):
        """Update the display of files based on newly generated augmented files."""
        for index, file_path in enumerate(all_augmented_filepaths):
            # Verify the file still exists before displaying it
            if os.path.exists(file_path):
                new_output_file_container = FileContainerWidget(file_path, self)
                new_output_file_container.hide_retry_button()
                new_output_file_container.hide_remove_button()

                # Check if this is the first file
                if index == 0:  # This means it's the first file
                    self.svc_preview.display_file_contents(file_path, 1)
                    self.svc_preview.display_graph_contents(file_path, 1)
                    self.svc_preview.display_emothaw_contents(file_path, 1)

        self.svc_preview.set_augmented_files(all_augmented_filepaths)

        # Automatically expand the output collapsible widget
        self.collapsible_widget_output.toggle_container(True)

    def update_original_absolute_file_display(self, original_absolute_files):
        """Update the display of original absolute files based on newly generated augmented files."""
        for index, file_path in enumerate(original_absolute_files):
            # Verify the file still exists before displaying it
            if os.path.exists(file_path):
                if index == 0:  # This means it's the first file
                    self.svc_preview.display_file_contents(file_path, 0)
                    self.svc_preview.display_graph_contents(file_path, 0)
                    self.svc_preview.display_emothaw_contents(file_path, 0)

        
        self.svc_preview.set_original_absolute_files(original_absolute_files)

    def add_more_files(self):
        self.file_upload_widget.open_file_dialog()

    def get_image_path(self, image_name):
        path = os.path.join(os.path.dirname(__file__), "..", "icon", image_name)
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
        self.model_widget.uncheck_checkbox()
        self.model_widget.slider_widget.resetValue()
        
        # Collapse all widgets except Input
        self.collapsible_widget_preview.toggle_container(False)
        self.collapsible_model_container.toggle_container(False)
        self.collapsible_widget_process_log.toggle_container(False)
        self.collapsible_widget_output.toggle_container(False)
        self.collapsible_widget_result.toggle_container(False)


if __name__ == "__main__":

    app = QtWidgets.QApplication([])
    window = Workplace()
    window.show()
    app.exec_()
