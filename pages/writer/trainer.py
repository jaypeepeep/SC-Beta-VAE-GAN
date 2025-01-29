import os
import time
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
import re
from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from PyQt5.QtCore import QThread, pyqtSignal
from model.scbetavaegan_pentab import (
    upload_and_process_files,
    process_dataframes,
    convert_and_store_dataframes,
    nested_augmentation,
    save_model,
    VAE,
    LSTMDiscriminator,
    train_models,
    ensure_data_compatibility,
    save_original_data,
    fill_gaps_and_interpolate
)

class ModelTrainingThread(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)
    zip_ready = pyqtSignal(str)
    partial_metric_ready = pyqtSignal(str, str)
    metrics_ready = pyqtSignal(dict)
    original_files_ready = pyqtSignal(list) 
    augmented_files_ready = pyqtSignal(list)

    def __init__(
        self,
        handwriting_dir,
        file_list,
        uploads_dir,
        selected_file,
        num_augmented_files,
        epochs=10,
        logger=None,
    ):
        super().__init__()
        self.uploads_dir = uploads_dir
        self.selected_file = selected_file
        self.num_augmented_files = (
            num_augmented_files
        )
        self.epochs = epochs
        self.logger = logger
        self.uploaded_files = file_list
        self.num_of_files = len(self.uploaded_files)
        self.handwriting_dir = handwriting_dir

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.folder_name = f"SyntheticData_{timestamp}"
        self.synthetic_data_dir = os.path.join(
            uploads_dir, self.folder_name
        )
        os.makedirs(self.synthetic_data_dir, exist_ok=True)

        self.model_output_dir = os.path.join("model", "pentab_vae_models")
        os.makedirs(self.model_output_dir, exist_ok=True)

        self.imputed_folder = os.path.abspath("files/imputed_handwriting")
        self.augmented_folder = os.path.abspath("files/augmented_data_handwriting")

    def run(self):
        self.log("Starting the process for file: " + self.handwriting_dir)

        # Step 1: Load only the selected .svc file from the uploads directory
        self.log(f"Using file path: {self.handwriting_dir}")
        try:
            (
                data_frames,
                processed_data,
                scalers,
                avg_data_points,
                input_filenames,
                original_data_frames,
            ) = upload_and_process_files(self.handwriting_dir, self.num_of_files)
            original_absolute_files = save_original_data(data_frames, input_filenames, "files/original_absolute_handwriting")
            self.original_files_ready.emit(original_absolute_files)
            
            self.log("File loaded and processed successfully.")
            self.log(f"Number of data frames loaded: {len(data_frames)}")
        except Exception as e:
            self.log(f"Error processing file: {e}", level="ERROR")
            self.finished.emit()
            return

        # Step 2: Process and save the loaded dataframes
        self.log("Converting and saving the processed dataframes...")
        data_frames = fill_gaps_and_interpolate(data_frames)
        convert_and_store_dataframes(input_filenames, data_frames)
        self.log("Data frames converted and saved.")

        self.log("Processing the data frames...")
        process_dataframes(data_frames)
        self.log("Processing of data frames completed.")

        # Step 3: Initialize the VAE model and LSTM Discriminator
        vae = VAE(latent_dim=512, beta=0.000001)
        lstm_discriminator = LSTMDiscriminator()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.log("VAE and LSTM Discriminator initialized.")

        # Step 4: Train the model with uploaded data
        self.log(f"Training started for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.log(f"Epoch {epoch + 1}/{self.epochs} in progress...")
            train_models(
                vae,
                lstm_discriminator,
                processed_data,
                original_data_frames,
                data_frames,
                num_augmented_files=self.num_augmented_files,
                epochs=1,
                optimizer=optimizer,
            )
            self.log(f"Epoch {epoch + 1} completed.")

        self.log("Training completed.")

        # Step 5: Save the trained model
        model_output_path = os.path.join(self.model_output_dir)
        model_file_path = os.path.join(self.model_output_dir, "final_vae_model.h5")
        save_model(vae, model_output_path)
        self.log(f"Model saved at {model_output_path}")

        # Step 6: Generate augmented data using nested_augmentation
        self.log("Generating nested augmented data...")
        try:
            augmented_datasets, self.all_augmented_path = nested_augmentation(
                num_augmentations=self.num_augmented_files,
                num_files_to_use=len(processed_data),
                data_frames=data_frames,
                scalers=scalers,
                input_filenames=input_filenames,
                original_data_frames=original_data_frames,
                model_path=model_file_path,
                avg_data_points=avg_data_points,
                processed_data=processed_data,
            )
            self.augmented_files_ready.emit(self.all_augmented_path)
            if augmented_datasets is None:
                self.log("nested_augmentation returned None.", level="ERROR")
                self.finished.emit()
                return
            self.log("Nested synthetic data generation completed.")
        except Exception as e:
            self.log(f"Error during nested augmentation: {e}", level="ERROR")
            self.finished.emit()
            return

        # Step 7: Zip the synthetic data files
        base_name = os.path.splitext(self.selected_file)[0]
        matching_files = self.get_matching_synthetic_files(base_name)

        if matching_files:
            zip_file_path = self.create_zip(matching_files)
            self.log(f"Zipped synthetic data saved at {zip_file_path}")
            self.zip_ready.emit(zip_file_path)
        else:
            self.log("No matching synthetic files found to zip.", level="WARNING")

        self.zip_ready.emit(zip_file_path)

        # Step 8: Directly Load and Compare Files for Metrics
        self.log("Calculating metrics for generated synthetic data...")
        metrics = {}

        # --- Embedded Functions for File Loading and Metrics ---
        def read_svc_file(file_path):
            """Log file reading and read SVC file data."""
            print(f"Reading file: {file_path}")
            return pd.read_csv(file_path, sep=' ', header=None, names=['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude'])
        
        def calculate_nrmse(original, predicted):
            """Calculate NRMSE between original and predicted datasets."""
            if original.shape != predicted.shape:
                raise ValueError("The shapes of the original and predicted datasets must match.")
            mse = np.mean((original - predicted) ** 2)
            rmse = np.sqrt(mse)
            nrmse = rmse / (np.max(original) - np.min(original))
            return nrmse
        
        def get_matching_augmented_files(original_file_path, augmented_folder):
            """Get matching augmented files based on original file names."""
            base_name = os.path.basename(original_file_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            print(f"Finding matching augmented files for: {base_name_without_ext}")

            # Update pattern to match augmented file naming correctly
            pattern = os.path.join(augmented_folder, f"synthetic_{base_name_without_ext}*.svc")
            matching_files = glob(pattern)

            # Log the matched files
            if matching_files:
                print(f"Matched files: {matching_files}")
            else:
                print(f"No matching augmented files found for: {base_name_without_ext}")

            def sort_key(filename):
                match = re.search(r'\((\d+)\)', filename)
                return int(match.group(1)) if match else -1
            
            return sorted(matching_files, key=sort_key)

        def calculate_nrmse_for_augmented_data(original_data_frames, augmented_data_list):
            """Calculate NRMSE for a list of original and augmented datasets."""
            nrmse_values = []

            for i, (original_df, augmented) in enumerate(zip(original_data_frames, augmented_data_list)):
                print(f"Processing original dataset {i + 1} and its corresponding augmented data.")
                original_array = original_df[['x', 'y', 'timestamp', 'pen_status']].values

                if isinstance(augmented, pd.DataFrame):
                    augmented = augmented.values
                elif not isinstance(augmented, np.ndarray):
                    raise ValueError(f"Unexpected data type for augmented data: {type(augmented)}")

                if augmented.shape[1] < 4:
                    raise ValueError(f"Augmented data has fewer than 4 columns: {augmented.shape}")

                augmented_array = augmented[:, :4]
                original_array, augmented_array = ensure_data_compatibility(original_array, augmented_array)

                try:
                    nrmse = calculate_nrmse(original_array, augmented_array)
                    nrmse_values.append(nrmse)
                    print(f"NRMSE for dataset {i + 1}: {nrmse:.4f}")
                except ValueError as e:
                    print(f"Error calculating NRMSE for dataset {i + 1}: {e}")

            average_nrmse = np.mean(nrmse_values) if nrmse_values else float('nan')
            print(f"Average NRMSE: {average_nrmse:.4f}")
            return nrmse_values, average_nrmse

        def create_lstm_classifier(input_shape):
            """Create and compile an LSTM model."""
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        
        def prepare_data_for_lstm(real_data, synthetic_data):
            """Prepare real and synthetic data for LSTM input."""
            n_features = min(real_data.shape[1], synthetic_data.shape[1])
            real_data_trimmed = real_data[:, :n_features]
            synthetic_data_trimmed = synthetic_data[:, :n_features]
            X = np.vstack((real_data_trimmed, synthetic_data_trimmed))
            y = np.concatenate((np.ones(len(real_data)), np.zeros(len(synthetic_data))))
            return X, y

        def post_hoc_discriminative_score(real_data, synthetic_data, n_splits=10):
            """Calculate the post-hoc discriminative score using K-Fold cross-validation."""
            X, y = prepare_data_for_lstm(real_data, synthetic_data)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            accuracies = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                model = create_lstm_classifier((1, X_train.shape[2]))
                model.fit(X_train, y_train, epochs=3, batch_size=256, verbose=0)
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            print(f"Post-Hoc Discriminative Score: Mean Accuracy = {mean_accuracy:.4f}, Std = {std_accuracy:.4f}")
            return mean_accuracy, std_accuracy
        
        def prepare_data(df, time_steps=5):
            """Prepare the data for LSTM input by creating sequences of specified length."""
            data = df[['x', 'y']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data)

            X, y = [], []
            for i in range(len(data_scaled) - time_steps):
                X.append(data_scaled[i:i + time_steps])
                y.append(data_scaled[i + time_steps])
            return np.array(X), np.array(y), scaler
        
        def create_model(input_shape):
            """Create and compile an LSTM model."""
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(50))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(2))
            model.compile(optimizer='adam', loss='mse')
            return model
        
        def evaluate_model(model, X_test, y_test, scaler):
            """Evaluate the model using MAPE."""
            y_pred = model.predict(X_test)
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform(y_test)
            mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
            print(f"MAPE: {mape * 100:.2f}%")
            return mape

        def k_fold_cross_validation(X, y, scaler, n_splits=10):
            """Perform K-Fold cross-validation on the LSTM model and return mean and std of MAPE."""
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(1000))
            mape_values = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = create_model((X_train.shape[1], X_train.shape[2]))
                model.fit(X_train, y_train, epochs=5, batch_size=512, verbose=0)
                mape = evaluate_model(model, X_test, y_test, scaler)
                mape_values.append(mape)

            mean_mape = np.mean(mape_values)
            std_mape = np.std(mape_values)
            print(f"Mean MAPE: {mean_mape * 100:.2f}%")
            print(f"Standard Deviation of MAPE: {std_mape * 100:.2f}%")
            return mean_mape, std_mape

        try:
            self.log("Loading original data for metrics comparison...")
            original_file_paths = [os.path.join(self.imputed_folder, f) for f in input_filenames]
            original_data = [read_svc_file(file_path) for file_path in original_file_paths]
            self.log("Loading augmented data for metrics comparison...")
            augmented_files = []
            for original_file_path in original_file_paths:
                matching_files = get_matching_augmented_files(original_file_path, self.augmented_folder)
                augmented_files.extend(matching_files)
            augmented_data = [read_svc_file(aug_file) for aug_file in augmented_files]

            nrmse_values, average_nrmse = calculate_nrmse_for_augmented_data(original_data, augmented_data)
            metrics["Normalized Root Mean Square Error (NRMSE)"] = average_nrmse
            self.log(f"Average NRMSE: {average_nrmse:.4f}")

            real_data, synthetic_data = np.concatenate(original_data), np.concatenate(augmented_data)
            mean_acc, std_acc = post_hoc_discriminative_score(real_data, synthetic_data)
            metrics["Discriminative Mean Accuracy"] = mean_acc
            metrics["Discriminative Accuracy Std"] = std_acc

            X, y, scaler = prepare_data(data_frames[0])
            mean_mape, std_mape = k_fold_cross_validation(X, y, scaler)
            metrics["Mean MAPE"] = mean_mape
            metrics["Standard Deviation of MAPE"] = std_mape

        except Exception as e:
            self.log(f"Error calculating NRMSE: {e}", level="ERROR")
            metrics["Average NRMSE"] = "Error"
        self.metrics_ready.emit(metrics)
        self.finished.emit()

    def get_matching_synthetic_files(self, base_name):
        """Find synthetic files matching the base name in the augmented folder."""
        pattern = os.path.join(self.augmented_folder, f"synthetic_{base_name}*.svc")
        matching_files = glob(pattern)
        return matching_files
    
    def create_zip(self, files):
        """Create a zip file from the provided list of files."""
        zip_file_path = os.path.join(self.synthetic_data_dir + ".zip")
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for file in files:
                zipf.write(file, os.path.basename(file))
        return zip_file_path

    def log(self, message, level="INFO"):
        if self.logger:
            if level == "ERROR":
                self.logger.error(message)
            else:
                self.logger.info(message)
        if self.log_signal:
            self.log_signal.emit(message)

