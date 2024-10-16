import subprocess
import requests
import os
import sys
import time
import shutil
import tempfile
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
import re
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QVBoxLayout, QScrollArea, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from components.button.handwriting_button import handwritingButton
from components.widget.collapsible_widget import CollapsibleWidget
from components.widget.file_preview_widget import FilePreviewWidget
from components.widget.process_log_widget import ProcessLogWidget
from components.widget.output_widget import OutputWidget
from components.widget.file_container_widget import FileContainerWidget
from components.widget.plot_container_widget import PlotContainerWidget
from components.widget.spin_box_widget import SpinBoxWidget
from components.widget.result_preview_widget import SVCpreview
from model.scbetavaegan_pentab import (
    upload_and_process_files,
    process_dataframes,
    convert_and_store_dataframes,
    nested_augmentation,
    save_model,
    download_augmented_data_with_modified_timestamp,
    VAE,
    LSTMDiscriminator,
    train_models,
    calculate_nrmse,
    post_hoc_discriminative_score,
<<<<<<< HEAD
    calculate_nrmse_for_augmented_data,
    k_fold_cross_validation,
    prepare_data_for_lstm,
    ensure_data_compatibility,
    save_original_data,
=======
    ensure_data_compatibility
>>>>>>> origin/nichole-branch
)


class ModelTrainingThread(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)
    zip_ready = pyqtSignal(str)
    partial_metric_ready = pyqtSignal(str, str)
    metrics_ready = pyqtSignal(dict)
    original_files_ready = pyqtSignal(list) 

    def __init__(
        self,
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
            num_augmented_files  # This is passed to nested_augmentation
        )
        self.epochs = epochs
        self.logger = logger
        self.uploaded_files = file_list
        self.num_of_files = len(self.uploaded_files)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.synthetic_data_dir = os.path.join(
            uploads_dir, f"SyntheticData_{timestamp}"
        )
        os.makedirs(self.synthetic_data_dir, exist_ok=True)

        self.model_output_dir = os.path.join("model", "pentab_vae_models")
        os.makedirs(self.model_output_dir, exist_ok=True)

        self.imputed_folder = os.path.abspath("imputed")
        self.augmented_folder = os.path.abspath("augmented_data")

    def run(self):
        self.log("Starting the process for file: " + self.selected_file)

        # Step 1: Load only the selected .svc file from the uploads directory
        file_path = os.path.join(self.uploads_dir, self.selected_file)
        self.log(f"Using file path: {file_path}")
        try:
            (
                data_frames,
                processed_data,
                scalers,
                avg_data_points,
                input_filenames,
                original_data_frames,
            ) = upload_and_process_files(file_path, self.num_of_files)
            original_absolute_files = save_original_data(data_frames, input_filenames)
            self.original_files_ready.emit(original_absolute_files)
            
            self.log("File loaded and processed successfully.")
            self.log(f"Number of data frames loaded: {len(data_frames)}")
        except Exception as e:
            self.log(f"Error processing file: {e}", level="ERROR")
            self.finished.emit()
            return

        # Step 2: Process and save the loaded dataframes
        self.log("Converting and saving the processed dataframes...")
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
            augmented_datasets = nested_augmentation(
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
            if augmented_datasets is None:
                self.log("nested_augmentation returned None.", level="ERROR")
                self.finished.emit()
                return
            self.log("Nested synthetic data generation completed.")
        except Exception as e:
            self.log(f"Error during nested augmentation: {e}", level="ERROR")
            self.finished.emit()
            return

        # Step 7: Save augmented data as .svc files
        download_augmented_data_with_modified_timestamp(
            augmented_datasets,
            scalers,
            original_data_frames,
            input_filenames,
            self.synthetic_data_dir,
        )
        self.log(f"Synthetic data saved in {self.synthetic_data_dir}")

        # Step 8: Zip the synthetic data files
        zip_file_path = self.create_zip(self.synthetic_data_dir)
        self.log(f"Zipped synthetic data saved at {zip_file_path}")

        self.zip_ready.emit(zip_file_path)

        # Step 8: Directly Load and Compare Files for Metrics
        self.log("Calculating metrics for generated synthetic data...")
        metrics = {}

<<<<<<< HEAD
        # Convert synthetic data if needed
        augmented_datasets = [
            np.array(data) if isinstance(data, list) else data
            for data in augmented_datasets
        ]
=======
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

                # Ensure augmented is a NumPy array and check its shape
                if isinstance(augmented, pd.DataFrame):
                    augmented = augmented.values
                elif not isinstance(augmented, np.ndarray):
                    raise ValueError(f"Unexpected data type for augmented data: {type(augmented)}")

                # Ensure augmented has at least 4 columns
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
            model.add(tf.keras.layers.Dropout(0.2))  # Adding dropout to introduce randomness
            model.add(tf.keras.layers.LSTM(50))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(2))  # Predict x and y
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
>>>>>>> origin/nichole-branch

        try:
<<<<<<< HEAD
            nrmse_values, average_nrmse = calculate_nrmse_for_augmented_data(
                original_data_frames, augmented_datasets
            )
            metrics["Average NRMSE"] = average_nrmse
=======
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
>>>>>>> origin/nichole-branch
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
<<<<<<< HEAD

        try:
            original_array, synthetic_array = ensure_data_compatibility(
                original_data_frames[0].values, augmented_datasets[0]
            )
            discriminative_mean, discriminative_std = post_hoc_discriminative_score(
                original_array, synthetic_array
            )
            metrics["Discriminative Score Mean"] = discriminative_mean
            metrics["Discriminative Score Std"] = discriminative_std
            self.log(
                f"Discriminative Score: Mean = {discriminative_mean:.4f}, Std = {discriminative_std:.4f}"
            )
        except Exception as e:
            self.log(f"Error calculating Discriminative Score: {e}", level="ERROR")
            metrics["Discriminative Score"] = "Error"

        try:
            X, y = prepare_data_for_lstm(
                original_data_frames[0].values, augmented_datasets[0]
            )
            mean_mape, std_mape = k_fold_cross_validation(X, y, scalers[0], n_splits=10)
            metrics["Predictive Score Mean MAPE"] = mean_mape * 100
            metrics["Predictive Score Std MAPE"] = std_mape * 100
            self.log(
                f"Predictive Score (MAPE): Mean = {mean_mape * 100:.2f}%, Std = {std_mape * 100:.2f}%"
            )
        except Exception as e:
            self.log(f"Error calculating Predictive Score (MAPE): {e}", level="ERROR")
            metrics["Predictive Score"] = "Error"

        # Emit the metrics
=======
>>>>>>> origin/nichole-branch
        self.metrics_ready.emit(metrics)
        # Notify completion
        self.finished.emit()

    def create_zip(self, directory):
        """Create a zip file from the generated synthetic data."""
        zip_file_path = os.path.join(directory + ".zip")
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for root, _, files in os.walk(directory):
                for file in files:
                    zipf.write(os.path.join(root, file), file)
        return zip_file_path

    def log(self, message, level="INFO"):
        if self.logger:
            if level == "ERROR":
                self.logger.error(message)
            else:
                self.logger.info(message)
        if self.log_signal:
            self.log_signal.emit(message)


class Handwriting(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Handwriting, self).__init__(parent)
        self.drawing_done = False
        self.flask_process = None
        self.file_list = []  # List to store uploaded .svc files
        self.uploads_dir = os.path.abspath("uploads")
        self.threads = []
        self.setupUi()

        if not os.path.exists(self.uploads_dir):
            os.makedirs(self.uploads_dir)

    def setupUi(self):
        """Initial setup for the drawing page or Flask app depending on the file_list state."""
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setContentsMargins(50, 0, 50, 50)

        # Initialize Process Log Widget
        self.process_log_widget = ProcessLogWidget(self)
        self.logger = self.process_log_widget.get_logger()
        self.layout.addWidget(self.process_log_widget)

        # Initialize Output Widget
        self.output_widget = OutputWidget(self)
        self.layout.addWidget(self.output_widget)

        # Initialize Result Preview Widget
        self.result_preview_widget = SVCpreview(self)
        self.layout.addWidget(self.result_preview_widget)

        # Set widgets initially collapsed
        self.process_log_widget.setVisible(False)
        self.output_widget.setVisible(False)
        self.result_preview_widget.setVisible(False)

        # Check if there is existing handwriting data (i.e., file_list is not empty)
        if self.file_list:
            # If handwriting data exists, skip the drawing page and show the embedded browser
            self.show_embedded_browser()
        else:
            # If no handwriting data, show the drawing page with the button
            self.show_drawing_page()

    def clear_layout(self):
        """Clear the current layout and any child layouts."""
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset references to deleted widgets
        self.process_log_widget = None
        self.result_preview_widget = None
        self.output_widget = None

    def clear_layout_recursively(self, layout):
        """Recursively clear all widgets and child layouts in the given layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            if item.layout():
                self.clear_layout_recursively(item.layout())
            del item

    def show_drawing_page(self):
        """Show the drawing page layout with the Draw and Handwrite button."""
        self.clear_layout()

        # Create a layout for the text
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.setAlignment(QtCore.Qt.AlignCenter)
        top_layout.setContentsMargins(0, 20, 0, 20)

        # Add text
        top_text = QtWidgets.QLabel("Draw and Handwrite", self)
        top_text.setAlignment(QtCore.Qt.AlignCenter)
        top_text.setStyleSheet("font-size: 30px; font-weight: bold; color: #033; ")
        top_layout.addWidget(top_text)
        self.layout.addLayout(top_layout)

        # Create and add the handwriting button
        drawButton = handwritingButton(self)
        drawButton.setContentsMargins(50, 20, 50, 50)
        self.layout.addWidget(drawButton)

        # Connect the button's click events
        drawButton.clicked.connect(self.show_confirmation_dialog)

    def show_confirmation_dialog(self):
        """Show a confirmation dialog before proceeding to the drawing page."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Proceed to Handwriting & Drawing")
        message_box.setText("Do you want to start drawing and handwriting?")
        message_box.setStandardButtons(
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        message_box.setDefaultButton(QtWidgets.QMessageBox.Ok)

        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Ok:
            self.run_flask_app()

    def run_flask_app(self):
        """Run the Flask app located in components/canvas/app.py and open it in the embedded browser."""
        flask_app_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../components/canvas/app.py")
        )

        # Run the Flask app as a subprocess
        self.flask_process = subprocess.Popen(["python", flask_app_path])

        # Display the embedded browser after a short delay to ensure Flask is running
        QtCore.QTimer.singleShot(5000, self.show_embedded_browser)

    def show_embedded_browser(self):
        """Show the Flask app inside the Handwriting page using QWebEngineView."""
        self.clear_layout()

        # Create a QWebEngineView and load the Flask app's URL
        self.webview = QWebEngineView(self)
        self.webview.setUrl(QtCore.QUrl("http://127.0.0.1:5000"))
        self.layout.addWidget(self.webview)

        # Ensure the webview resizes responsively
        self.webview.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        # Poll Flask to check if drawing is done and file is uploaded
        QtCore.QTimer.singleShot(
            5000, self.check_drawing_done
        )  # Adjust the delay if necessary

    def check_drawing_done(self):
        """Periodically check if the drawing is done by querying Flask."""
        try:
            response = requests.get("http://127.0.0.1:5000/check_upload")
            if response.status_code == 200:
                data = response.json()
                filename = data.get("filename")
                if filename.endswith(".svc"):  # Ensure file is an .svc file
                    self.show_done_page(filename)
                    self.svc_preview.set_uploaded_files(self.file_list)
                    if filename not in self.file_list:  # Avoid duplicate
                        self.file_list.append(filename)
                        print("File list:", self.file_list)
                        if hasattr(self, "file_preview_widget"):
                            self.file_preview_widget.set_uploaded_files(self.file_list)
                else:
                    self.process_log_widget.append_log(f"Invalid file type: {filename}")
            else:
                QtCore.QTimer.singleShot(5000, self.check_drawing_done)
        except requests.ConnectionError:
            QtCore.QTimer.singleShot(5000, self.check_drawing_done)

    def show_done_page(self, filename):
        """Show the page after the drawing is completed."""
        self.file_list.append(filename)  # Append the new filename to the list
        self.clear_layout()

        # Create a scroll area to wrap the collapsible content
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        # Create a widget that will be placed inside the scroll area
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setAlignment(QtCore.Qt.AlignTop)

        # Create a scrollable widget
        sub_area = QScrollArea()
        sub_area.setWidgetResizable(True)

        # Create a container for the scroll area
        sub_container = QWidget()
        sub_container.setMaximumHeight(300)
        sub_layout = QVBoxLayout(sub_container)

        # Add file containers to the scrollable layout
        for file in self.file_list:
            file_container = FileContainerWidget(file, self)
            sub_layout.addWidget(file_container)

        # Set the scrollable widget
        sub_area.setWidget(sub_container)

        # Add the scroll area to the collapsible widget
        ##
        # Add the scroll area to the main layout
        scroll_area.setWidget(scroll_widget)
        self.layout.addWidget(scroll_area)

        # Call the collapsible widget component
        self.collapsible_widget = CollapsibleWidget("Input", self)
        scroll_layout.addWidget(self.collapsible_widget)
        self.collapsible_widget.toggle_container(True)

        # Add a file container widget to the collapsible widget for each drawing added
        for file in self.file_list:
            file_container = FileContainerWidget(file, self)
            self.collapsible_widget.add_widget(file_container)

        # Add the dropdown (QComboBox) for selecting a file to plot
        self.file_dropdown = QtWidgets.QComboBox(self)
        self.file_dropdown.setStyleSheet(
            """
            QComboBox {
                background-color: #033;  
                color: white; 
                font-weight: bold;           
                font-family: Montserrat; 
                font-size: 14px;        
                padding: 10px;            
                border: 2px solid #033;  
                border-radius: 5px;      
            }

            /* Dropdown arrow styling */
            QComboBox::drop-down {
                border: none;
            }

            /* Dropdown arrow icon */
            QComboBox::down-arrow {
                image: url(arrow_down_icon.png); 
                width: 14px;
                height: 14px;
            }

            /* Styling for the dropdown items */
            QComboBox QAbstractItemView {
                background-color: white;   
                color: #033;                 
                border: 1px solid #033;    
                font-family: Montserrat;
                font-size: 14px;
        }"""
        )
        self.file_dropdown.addItems(self.file_list)
        self.file_dropdown.currentIndexChanged.connect(self.on_file_selected)

        # Add the dropdown to the collapsible widget
        self.collapsible_widget.add_widget(self.file_dropdown)

        # Add the plot container widget
        self.plot_container = PlotContainerWidget(self)
        self.collapsible_widget.add_widget(self.plot_container)

        # Initially load the plot for the first file in the list
        if self.file_list:
            self.plot_container.loadPlot(self.file_list[0])

        # Add the slider widget directly to the collapsible widget
        self.spin_box_widget = SpinBoxWidget(1)
        self.collapsible_widget.add_widget(self.spin_box_widget)

        # Add "Draw More" and "Clear All" buttons inside the collapsible widget
        button_layout = QtWidgets.QHBoxLayout()

        self.draw_more_button = QtWidgets.QPushButton("Draw More", self)
        self.draw_more_button.setStyleSheet(
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
        self.draw_more_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.draw_more_button.clicked.connect(self.run_flask_app)

        self.clear_all_button = QtWidgets.QPushButton("Clear All", self)
        self.clear_all_button.setStyleSheet(
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
        self.clear_all_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.clear_all_button.clicked.connect(self.clear_all_drawings)

        # Add the buttons to the button layout
        button_layout.addWidget(self.draw_more_button)
        button_layout.addWidget(self.clear_all_button)

        # Add the button layout to the collapsible widget
        button_widget = QtWidgets.QWidget()  # Wrap buttons in a QWidget
        button_widget.setLayout(button_layout)
        self.collapsible_widget.add_widget(button_widget)

        # Add the File Preview Widget
        self.collapsible_widget_file_preview = CollapsibleWidget("File Preview", self)
        scroll_layout.addWidget(self.collapsible_widget_file_preview)
        self.file_preview_widget = FilePreviewWidget(self)
        self.file_preview_widget.set_uploaded_files(self.file_list)
        self.collapsible_widget_file_preview.add_widget(self.file_preview_widget)

        # Add the Process Log Widget
        self.collapsible_widget_process_log = CollapsibleWidget("Process Log", self)
        scroll_layout.addWidget(self.collapsible_widget_process_log)
        self.process_log_widget = ProcessLogWidget(self)
        self.collapsible_widget_process_log.add_widget(self.process_log_widget)

        # Add the Output Widget
        self.collapsible_widget_output = CollapsibleWidget("Output", self)
        scroll_layout.addWidget(self.collapsible_widget_output)
        self.output_widget = OutputWidget(self)
        self.collapsible_widget_output.add_widget(self.output_widget)

        # Call the collapsible widget component for result
        self.collapsible_widget_result = CollapsibleWidget("Result", self)
        scroll_layout.addWidget(self.collapsible_widget_result)
        self.svc_preview = SVCpreview(input=filename)
        self.collapsible_widget_result.add_widget(self.svc_preview)

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
        self.generate_data_button.setFixedSize(250, 50)
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
        scroll_layout.addLayout(button_layout)

        # Automatically open file preview widget after 2 secs
        QTimer.singleShot(
            2000, lambda: self.collapsible_widget_file_preview.toggle_container(True)
        )

    def on_generate_data(self):
        """Start processing the selected .svc files."""
        uploads_dir = "uploads"
        num_augmented_files = self.spin_box_widget.number_input.value()
        epochs = 10

        if not self.file_list:
            self.process_log_widget.append_log("No files available for processing.")
            return

        self.process_log_widget.setVisible(True)
        self.collapsible_widget_process_log.toggle_container(True)
        self.generate_data_button.setEnabled(False)

        file_count = len(self.file_list)
        self.process_log_widget.append_log(
            f"Starting data generation for {file_count} file(s)..."
        )

        for selected_file in self.file_list:
            if not selected_file.endswith(".svc"):
                self.process_log_widget.append_log(
                    f"Skipping non-.svc file: {selected_file}"
                )
                continue

            # Start a new thread for each file
            thread = ModelTrainingThread(
                self.file_list,
                uploads_dir,
                selected_file,
                num_augmented_files,
                epochs,
                logger=self.logger,
            )
            self.threads.append(thread)  # Keep track of threads
            thread.log_signal.connect(self.process_log_widget.append_log)
            thread.zip_ready.connect(self.on_zip_ready)
            thread.metrics_ready.connect(self.on_metrics_ready)
            thread.finished.connect(self.on_thread_finished)
            thread.original_files_ready.connect(self.update_original_absolute_file_display)  # Connect the new signal
            thread.start()

        self.process_log_widget.append_log("All threads started, awaiting results...")

    def closeEvent(self, event):
        """Ensure the Flask app process and threads are killed when the main window is closed."""
        # Terminate the Flask process if running
        if self.flask_process:
            self.flask_process.terminate()

        # Stop all running threads
        for thread in self.threads:
            if thread.isRunning():
                thread.quit()  # Stop the thread
                thread.wait()  # Wait until it's fully terminated

        event.accept()

    def on_thread_finished(self):
        """Callback when a single file has finished processing."""
        self.process_log_widget.append_log("A file has finished processing.")

        # Check if all threads are done before re-enabling the button
        for thread in self.threads:
            if thread.isFinished():
                self.threads.remove(thread)  # Remove finished threads

        if not self.threads:  # If all threads are finished
            self.process_log_widget.append_log("All files have finished processing.")
            self.generate_data_button.setEnabled(True)

    def on_zip_ready(self, zip_file_path):
        # Set the zip path for output widget
        if hasattr(self.output_widget, "set_zip_path"):
            QtCore.QMetaObject.invokeMethod(
                self.output_widget,
                "set_zip_path",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, zip_file_path),
            )
            self.output_widget.setVisible(True)
            self.collapsible_widget_output.toggle_container(True)

        self.svc_preview.add_graph_containers()
        self.update_output_file_display(zip_file_path)
        self.collapsible_widget_result.toggle_container(True)

    def on_metrics_ready(self, metrics):
        """Update the results_text widget with the calculated metrics."""
<<<<<<< HEAD
        metrics_text = "Calculated Metrics:\n"
        for key, value in metrics.items():
            metrics_text += f"{key}: {value}\n"

=======
        # Start building the formatted text for results
        metrics_text = ""

        # Normalized Root Mean Square Error (NRMSE)
        if "Normalized Root Mean Square Error (NRMSE)" in metrics:
            overall_avg_nrmse = metrics["Normalized Root Mean Square Error (NRMSE)"]
            metrics_text += "Normalized Root Mean Square Error (NRMSE)\n"
            metrics_text += f"\tOverall Average NRMSE: {overall_avg_nrmse:.4f}\n\n"

        # Post-Hoc Discriminative Score (PHDS)
        if "Discriminative Mean Accuracy" in metrics and "Discriminative Accuracy Std" in metrics:
            mean_acc = metrics["Discriminative Mean Accuracy"]
            std_acc = metrics["Discriminative Accuracy Std"]
            metrics_text += "Post-Hoc Discriminative Score (PHDS)\n"
            metrics_text += f"\tMean accuracy: {mean_acc:.4f} (±{std_acc:.4f})\n\n"

        # Post-Hoc Predictive Score (PHPS)
        if "Mean MAPE" in metrics and "Standard Deviation of MAPE" in metrics:
            mean_mape = metrics["Mean MAPE"] * 100  # Convert to percentage
            std_mape = metrics["Standard Deviation of MAPE"] * 100  # Convert to percentage
            metrics_text += "Post-Hoc Predictive Score (PHPS)\n"
            metrics_text += f"\tMean MAPE: {mean_mape:.2f}%\n"
            metrics_text += f"\tStandard Deviation of MAPE: {std_mape:.2f}%\n"

        # Update the text in the results preview widget
>>>>>>> origin/nichole-branch
        self.svc_preview.results_text.setPlainText(metrics_text)


    def on_training_finished(self):
        """Callback when training and data generation is finished."""
        self.generate_data_button.setText("Generate Synthetic Data")
        self.generate_data_button.setEnabled(True)
        self.process_log_widget.append_log("Data generation finished.")

    def get_absolute_paths(self, directory, filenames):
        """
        Given a directory and a list of filenames, return a list of absolute paths.

        Args:
            directory (str): The base directory where the files are located.
            filenames (list): A list of filenames (relative paths).

        Returns:
            list: A list of absolute paths.
        """
        absolute_paths = []
        for filename in filenames:
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            absolute_paths.append(absolute_path)
        return absolute_paths

    def extract_paths_from_zip(self, zip_path, extract_to):
        """
        Extract the .svc files from a zip archive and return their absolute paths.

        Args:
            zip_path (str): Path to the zip file containing synthetic data.
            extract_to (str): Directory where the files will be extracted.

        Returns:
            list: A list of absolute paths to the extracted .svc files.
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract all .svc files to the specified directory
            zip_ref.extractall(extract_to)

        # Gather paths of all extracted .svc files
        svc_paths = [
            os.path.abspath(os.path.join(extract_to, file))
            for file in os.listdir(extract_to)
            if file.endswith(".svc")
        ]
        return svc_paths

    def update_output_file_display(self, zip_file_path):
        """
        Update the display of files based on newly generated augmented files.
        """
        # Create a unique directory based on the current timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        synthetic_output_dir = os.path.join(
            "extracted_synthetic_data", f"run_{timestamp}"
        )

        # Ensure the directory exists
        os.makedirs(synthetic_output_dir, exist_ok=True)

        # Extract paths from the zip file using the new function
        synthetic_paths = self.extract_paths_from_zip(
            zip_file_path, synthetic_output_dir
        )

        # Ensure paths are correctly set and the files exist
        for index, file_path in enumerate(synthetic_paths):
            if os.path.exists(file_path):
                if index == 0:  # Display the first file
                    self.svc_preview.display_file_contents(file_path, 1)
                    self.svc_preview.display_graph_contents(file_path, 1)
                    self.svc_preview.display_handwriting_contents(file_path, 1)

        self.svc_preview.set_augmented_files(synthetic_paths)

        # Automatically expand the output collapsible widget
        self.collapsible_widget_output.toggle_container(True)

    def update_original_absolute_file_display(self, original_absolute_files):
        """Update the display of original absolute files based on newly generated augmented files."""
        for index, file_path in enumerate(original_absolute_files):
            if os.path.exists(file_path):
                if index == 0:  # This means it's the first file
                    self.svc_preview.display_file_contents(file_path, 0)
                    self.svc_preview.display_graph_contents(file_path, 0)
                    self.svc_preview.display_emothaw_contents(file_path, 0)

        self.svc_preview.set_original_absolute_files(original_absolute_files)

    def calculate_metrics(self, original_file, synthetic_file):
        """Calculate and return the NRMSE, discriminative, and predictive scores."""
        original_data = pd.read_csv(
            original_file,
            sep=" ",
            names=[
                "x",
                "y",
                "timestamp",
                "pen_status",
                "pressure",
                "azimuth",
                "altitude",
            ],
        )
        synthetic_data = pd.read_csv(
            synthetic_file,
            sep=" ",
            names=[
                "x",
                "y",
                "timestamp",
                "pen_status",
                "pressure",
                "azimuth",
                "altitude",
            ],
        )

        # Compute NRMSE
        nrmse = calculate_nrmse(
            original_data[["x", "y"]].values, synthetic_data[["x", "y"]].values
        )

        # Compute Post-Hoc Discriminative Score (you can use the LSTM model for this)
        discriminative_score = post_hoc_discriminative_score(
            original_data, synthetic_data
        )

        # Compute Post-Hoc Predictive Score (LSTM-based predictive model)
        # predictive_score = post_hoc_predictive_score(original_data, synthetic_data)

        return {
            "nrmse": nrmse,
            "discriminative_score": discriminative_score,
            # "predictive_score": predictive_score
        }

    def show_reset_confirmation_dialog(self):
        """Show a confirmation dialog before resetting the state."""
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Question)
        message_box.setWindowTitle("Discard and Retry")
        message_box.setText(
            "Are you sure you want to discard your current handwriting and start over?"
        )
        message_box.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        message_box.setDefaultButton(QtWidgets.QMessageBox.No)

        message_box.setStyleSheet("QPushButton { font-size: 14px; }")

        response = message_box.exec_()

        if response == QtWidgets.QMessageBox.Yes:
            self.reset_state()

    def clear_all_drawings(self):
        """Clear all added files and reset the state."""
        self.file_list.clear()  # Empty the file list

        # Stop all running threads when clearing all drawings
        for thread in self.threads:
            if thread.isRunning():
                thread.quit()
                thread.wait()

        self.threads.clear()  # Clear the thread list after stopping them

        self.show_drawing_page()  # Go back to the initial drawing page

    def reset_state(self):
        """Reset the state and go back to the drawing page."""
        self.drawing_done = False
        self.file_list.clear()  # Clear file list when resetting
        self.show_drawing_page()

    def on_file_selected(self):
        """Update the plot when a different file is selected from the dropdown."""
        selected_file = self.file_dropdown.currentText()
        self.plot_container.loadPlot(selected_file)

    def closeEvent(self, event):
        """Ensure the Flask app process is killed when the main window is closed."""
        if self.flask_process:
            self.flask_process.terminate()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Handwriting()
    window.resize(800, 600)  # Adjust window size for the embedded browser
    window.show()
    sys.exit(app.exec_())
