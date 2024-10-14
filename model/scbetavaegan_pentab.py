import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import zipfile
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from keras.utils import custom_object_scope
from tensorflow.keras.models import load_model

# 1. Load and process the .svc file
def upload_and_process_files(path, num_files_to_use=None):
    """Upload and process .svc files for handwriting analysis."""
    
    # Check if the path is a directory or a single file
    if os.path.isdir(path):
        svc_files = [f for f in os.listdir(path) if f.endswith('.svc')]
        directory = path
    else:
        svc_files = [os.path.basename(path)]  # Only use the single file
        directory = os.path.dirname(path)  # Get the directory from the file path

    # If num_files_to_use is specified, only take that many files sequentially
    if num_files_to_use:
        svc_files = svc_files[:num_files_to_use]  # Take the first num_files_to_use files

    data_frames = []  # Processed data after scaling
    original_data_frames = []  # Save the original unscaled data
    scalers = []
    input_filenames = []  # List to store input filenames

    num_files = len(svc_files)
    fig, axs = plt.subplots(1, num_files, figsize=(6*num_files, 6), constrained_layout=True)
    if num_files == 1:
        axs = [axs]

    # Create the folder if it doesn't exist
    output_folder = 'original_absolute'
    os.makedirs(output_folder, exist_ok=True)

    for i, filename in enumerate(svc_files):
        file_path = os.path.join(directory, filename)
        input_filenames.append(filename)  # Store the filename
        print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path, skiprows=1, header=None, delim_whitespace=True)
        df.columns = ['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']
        
        # Modify timestamp to start from 0
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).round().astype(int)
        
        # Save the modified data to the 'original_absolute' folder
        save_path = os.path.join(output_folder, filename)
        df.to_csv(save_path, sep=' ', index=False, header=False)
        
        # Keep a copy of the original data before scaling
        original_data_frames.append(df.copy())  # Save the original unmodified data
        
        # Process the data for use in the model
        df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]] 
        data_frames.append(df)
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df[['x', 'y', 'timestamp']])
        scalers.append(scaler)

        # Plot with a 90-degree right rotation (swap x and y)
        on_paper = df[df['pen_status'] == 1]
        in_air = df[df['pen_status'] == 0]
        axs[i].scatter(on_paper['x'], -on_paper['y'], c='blue', s=1, alpha=0.7, label='On Paper')  # Swap x and y, negate y
        axs[i].scatter(in_air['x'], -in_air['y'], c='red', s=1, alpha=0.7, label='In Air')  # Swap x and y, negate y
        axs[i].set_title(f'Original Data {i + 1}')
        axs[i].set_xlabel('x')  # x-axis is 'x'
        axs[i].set_ylabel('-y')  # y-axis is '-y'
        axs[i].legend()
        axs[i].set_aspect('equal')

        # Print the first few rows of the timestamp column
        print(f"Modified timestamps for file {filename}:")
        print(df['timestamp'].head())
        print("\n")

    # Call gap filling and interpolation function
    data_frames = fill_gaps_and_interpolate(data_frames)

    # Update processed data for all DataFrames
    processed_data = [np.column_stack((scaler.transform(df[['x', 'y', 'timestamp']]), df['pen_status'].values)) 
                      for df, scaler in zip(data_frames, scalers)]
    avg_data_points = int(np.mean([df.shape[0] for df in data_frames]))

    print(f"data_frames type: {type(data_frames)}, length: {len(data_frames)}")
    print(f"scalers type: {type(scalers)}, length: {len(scalers)}")
    print(f"processed_data type: {type(processed_data)}, length: {len(processed_data)}")

    return data_frames, processed_data, scalers, avg_data_points, input_filenames, original_data_frames

def fill_gaps_and_interpolate(data_frames):
    """Fill gaps in the timestamp and interpolate NaN values."""
    for df_idx in range(len(data_frames)):
        df = data_frames[df_idx]

        # Ensure 'timestamp' is numeric and sorted
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df.sort_values('timestamp', inplace=True)

        # Calculate time differences and identify gaps
        df['time_diff'] = df['timestamp'].diff()
        gap_indices = df.index[df['time_diff'] > 30000].tolist()

        new_rows = []
        for idx in gap_indices:
            if idx + 1 < len(df):
                current_timestamp = df.at[idx, 'timestamp']
                next_timestamp = df.at[idx + 1, 'timestamp']
                num_fill_entries = (next_timestamp - current_timestamp) // 20000

                for i in range(1, num_fill_entries + 1):
                    new_timestamp = current_timestamp + i * 70
                    new_row = {
                        'x': np.nan,
                        'y': np.nan,
                        'timestamp': new_timestamp,
                        'pen_status': 0,
                        'azimuth': df.at[idx, 'azimuth'],
                        'altitude': df.at[idx, 'altitude'],
                        'pressure': df.at[idx, 'pressure']
                    }
                    new_rows.append(new_row)

        new_rows_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows_df], ignore_index=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Interpolate NaN values in 'x' and 'y'
        if df[['x', 'y']].isnull().any().any():
            df[['x', 'y']] = df[['x', 'y']].interpolate(method='linear')

        df.drop(columns=['time_diff'], inplace=True)
        data_frames[df_idx] = df

    return data_frames

def convert_and_store_dataframes(input_filenames, data_frames):
    """Convert numeric columns to integers and store processed DataFrames."""
    imputed_folder = 'imputed'
    os.makedirs(imputed_folder, exist_ok=True)
    processed_dataframes = []

    for input_filename, df in zip(input_filenames, data_frames):
        # Convert all numeric columns to integers
        df[['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']] = df[['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']].astype(int)

        #########Start
        # Save the processed DataFrame to the 'imputed' folder with the same input filename
        save_path = os.path.join(imputed_folder, input_filename)
        df.to_csv(save_path, sep=' ', index=False, header=False)  # Save without header and index
        ##########Emd
        # Append the processed DataFrame to the list
        processed_dataframes.append(df)

        print(f"Processed DataFrame saved as: {input_filename}")

    return processed_dataframes

def process_dataframes(dataframes, num_files_to_use=None):
    """Process loaded dataframes by normalizing and modifying the timestamp."""
    if num_files_to_use:
        dataframes = dataframes[:num_files_to_use]

    data_frames = []
    scalers = []

    for i, df in enumerate(dataframes):
        # Modify timestamp to start from 0
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).round().astype(int)
        
        data_frames.append(df)
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df[['x', 'y', 'timestamp']])
        scalers.append(scaler)

        # Print the first few rows of the timestamp column
        print(f"Modified timestamps for DataFrame {i + 1}:")
        print(df['timestamp'].head())
        print("\n")

    # Create processed data by transforming 'x', 'y', 'timestamp', and keeping 'pen_status'
    processed_data = [np.column_stack((scaler.transform(df[['x', 'y', 'timestamp']]), df['pen_status'].values)) 
                      for df, scaler in zip(data_frames, scalers)]
    
    avg_data_points = int(np.mean([df.shape[0] for df in data_frames]))

    # Return the modified data frames, processed data, and original filenames
    return data_frames, processed_data, scalers, avg_data_points, [f"DataFrame_{i+1}" for i in range(len(dataframes))]

# 2. VAE Model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(4,)),  # 4 for x, y, timestamp, pen_status
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(latent_dim * 2)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4)  # 4 for x, y, timestamp, pen_status
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        decoded = self.decoder(z)
        xy_timestamp = tf.sigmoid(decoded[:, :3])  # x, y, and timestamp
        pen_status = tf.sigmoid(decoded[:, 3])
        return tf.concat([xy_timestamp, tf.expand_dims(pen_status, -1)], axis=1)

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        config.pop('dtype', None)
        return cls(**config)

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'beta': self.beta
        })
        return config

# 3. LSTM Discriminator
class LSTMDiscriminator(tf.keras.Model):
    def __init__(self):
        super(LSTMDiscriminator, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 4)),  # LSTM for sequence learning
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

    def call(self, x):
        return self.model(x)

# 4. Function to compute VAE loss
def compute_loss(model, x):
    x_reconstructed, mean, logvar = model(x)
    reconstruction_loss_xy_timestamp = tf.reduce_mean(tf.keras.losses.mse(x[:, :3], x_reconstructed[:, :3]))
    reconstruction_loss_pen = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x[:, 3], x_reconstructed[:, 3]))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss_xy_timestamp + reconstruction_loss_pen, kl_loss, model.beta * kl_loss

# 5. Functions for training steps
@tf.function
def train_vae_step(model, x, optimizer, lstm_discriminator=None):
    """Train a single step of the VAE model."""
    with tf.GradientTape() as tape:
        x_reconstructed, mean, logvar = model(x)
        reconstruction_loss, kl_loss, total_kl_loss = compute_loss(model, x)
        
        # Add LSTM discriminator loss if available
        if lstm_discriminator is not None:
            real_predictions = lstm_discriminator(tf.expand_dims(x, axis=0))
            fake_predictions = lstm_discriminator(tf.expand_dims(x_reconstructed, axis=0))
            discriminator_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_predictions), real_predictions) +
                                                tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_predictions), fake_predictions))
            generator_loss = reconstruction_loss + total_kl_loss + 0.1 * discriminator_loss  # Adjust the weight as needed
        else:
            generator_loss = reconstruction_loss + total_kl_loss
    
    gradients = tape.gradient(generator_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return generator_loss, reconstruction_loss, kl_loss

@tf.function
def train_lstm_step(lstm_model, real_data, generated_data, optimizer):
    """Train a single step of the LSTM discriminator model."""
    with tf.GradientTape() as tape:
        real_predictions = lstm_model(real_data)
        generated_predictions = lstm_model(generated_data)
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_predictions), real_predictions)
        generated_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(generated_predictions), generated_predictions)
        total_loss = real_loss + generated_loss
    gradients = tape.gradient(total_loss, lstm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, lstm_model.trainable_variables))
    return total_loss

def train_models(vae, lstm_discriminator, processed_data, epochs=10, vae_epochs=200, lstm_interval=50, batch_size=512, learning_rate=0.001):
    """Train the VAE and LSTM models."""
    lstm_optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    train_datasets = [tf.data.Dataset.from_tensor_slices(data).shuffle(10000).batch(batch_size) for data in processed_data]

    generator_loss_history = []
    reconstruction_loss_history = []
    kl_loss_history = []
    nrmse_history = []

    save_dir = "pentab_vae_models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        generator_loss = 0 
        reconstruction_loss_sum = 0
        kl_loss_sum = 0
        num_batches = sum(len(dataset) for dataset in train_datasets)

        with tqdm(total=num_batches, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for dataset in train_datasets:
                for batch in dataset:
                    use_lstm = epoch >= vae_epochs and (epoch - vae_epochs) % lstm_interval == 0
                    generator_loss_batch, reconstruction_loss, kl_loss = train_vae_step(vae, batch, optimizer, lstm_discriminator if use_lstm else None)
                    generator_loss += generator_loss_batch
                    reconstruction_loss_sum += reconstruction_loss
                    kl_loss_sum += kl_loss
                    pbar.update(1)
                    pbar.set_postfix({'Generator Loss': float(generator_loss_batch), 'Reconstruction Loss': float(reconstruction_loss), 'KL Loss': float(kl_loss)})

        # Train LSTM every `lstm_interval` epochs after `vae_epochs`
        if epoch >= vae_epochs and (epoch - vae_epochs) % lstm_interval == 0:
            for data in processed_data:
                augmented_data = vae.decode(tf.random.normal(shape=(data.shape[0], latent_dim))).numpy()
                real_data = tf.expand_dims(data, axis=0)
                generated_data = tf.expand_dims(augmented_data, axis=0)
                lstm_loss = train_lstm_step(lstm_discriminator, real_data, generated_data, lstm_optimizer)
            print(f'LSTM training at epoch {epoch+1}: Discriminator Loss = {lstm_loss.numpy()}')

        avg_generator_loss = generator_loss / num_batches  # Update the average calculation
        avg_reconstruction_loss = reconstruction_loss_sum / num_batches
        avg_kl_loss = kl_loss_sum / num_batches

        generator_loss_history.append(avg_generator_loss)  # Update history list
        reconstruction_loss_history.append(avg_reconstruction_loss)
        kl_loss_history.append(avg_kl_loss)

        # Calculate NRMSE
        nrmse_sum = 0
        for data in processed_data:
            augmented_data = vae.decode(tf.random.normal(shape=(data.shape[0], latent_dim))).numpy()
            rmse = np.sqrt(mean_squared_error(data[:, :2], augmented_data[:, :2]))
            nrmse = rmse / (data[:, :2].max() - data[:, :2].min())
            nrmse_sum += nrmse
        
        nrmse_avg = nrmse_sum / len(processed_data)
        nrmse_history.append(nrmse_avg)

        print(f"Epoch {epoch+1}: Generator Loss = {avg_generator_loss:.6f}, Reconstruction Loss = {avg_reconstruction_loss:.6f}, KL Divergence Loss = {avg_kl_loss:.6f}")
        print(f"NRMSE = {nrmse_avg:.6f}")

    return generator_loss_history, reconstruction_loss_history, kl_loss_history, nrmse_history


def post_process_pen_status(pen_status, threshold=0.5, min_segment_length=5):
    """Smooth out rapid changes in pen status."""
    binary_pen_status = (pen_status > threshold).astype(int)
    
    # Smooth out rapid changes
    for i in range(len(binary_pen_status) - min_segment_length):
        if np.all(binary_pen_status[i:i+min_segment_length] == binary_pen_status[i]):
            binary_pen_status[i:i+min_segment_length] = binary_pen_status[i]
    
    return binary_pen_status

# 6. Generate augmented datasets
def generate_augmented_datasets(model, processed_data, data_frames, num_augmented_files, avg_data_points, base_latent_variability=100.0, latent_variability_range=(0.99, 1.01)):
    """Generate augmented datasets using the pretrained VAE model."""
    augmented_datasets = []
    num_input_files = len(processed_data)

    # Validate that processed_data is a list of arrays
    if not isinstance(processed_data, list):
        raise ValueError(f"Expected processed_data to be a list, but got {type(processed_data)}")

    for data in processed_data:
        if not isinstance(data, (np.ndarray, list)):
            raise ValueError(f"Expected each element in processed_data to be a NumPy array or list, but got {type(data)}")

    for i in range(num_augmented_files):
        selected_data = processed_data[i % num_input_files]

        # Validate selected_data again before proceeding
        if not isinstance(selected_data, (np.ndarray, list)):
            raise ValueError(f"Expected selected_data to be a NumPy array or list, but got {type(selected_data)}")

        original_data = data_frames[i % num_input_files]  # Use original unprocessed data
        pressure_azimuth_altitude = original_data[['pressure', 'azimuth', 'altitude']].values
        
        # Latent variability calculation
        latent_variability = base_latent_variability * np.random.uniform(latent_variability_range[0], latent_variability_range[1])
        
        # Encode and reparameterize
        mean, logvar = model.encode(tf.convert_to_tensor(selected_data, dtype=tf.float32))
        z = model.reparameterize(mean, logvar * latent_variability)
        
        # Decode the latent variable back to data
        augmented_data = model.decode(z).numpy()

        # Post-process pen status
        augmented_data[:, 3] = post_process_pen_status(augmented_data[:, 3])
        
        # Ensure timestamps are in sequence
        augmented_data[:, 2] = np.sort(augmented_data[:, 2])
        
        # Append the pressure, azimuth, and altitude columns from the original data
        augmented_data = np.column_stack((augmented_data, pressure_azimuth_altitude[:augmented_data.shape[0]]))
        
        augmented_datasets.append(augmented_data)

    return augmented_datasets

# 7. Load pretrained VAE model
def load_pretrained_vae(model_path):
    """Load a pretrained VAE model from the specified path."""
    with tf.keras.utils.custom_object_scope({'VAE': VAE}):
        model = tf.keras.models.load_model(model_path)
    print("Pretrained VAE model loaded.")
    return model

# 8. Visualization functions
def visualize_augmented_data(augmented_datasets, scalers, original_data_frames, axs):
    """Visualize augmented data after inverse scaling."""
    all_augmented_data = []  # List to store augmented datasets after scaling back

    for i, (augmented_data, scaler, original_df) in enumerate(zip(augmented_datasets, scalers, original_data_frames)):
        # Inverse transform the augmented data
        augmented_xyz = scaler.inverse_transform(augmented_data[:, :3])
        
        # Round to integers
        augmented_xyz_int = np.rint(augmented_xyz).astype(int)
        
        # Get pen status from augmented data
        pen_status = augmented_data[:, 3].astype(int)
        
        # Prepare pressure, azimuth, altitude data from original data
        original_paa = original_df[['pressure', 'azimuth', 'altitude']].values
        
        # If augmented data is longer, fill the original data by repeating values backwards
        if len(augmented_data) > len(original_paa):
            original_paa = repeat_backwards(original_paa, len(augmented_data))
        
        # Round pressure, azimuth, altitude to integers
        original_paa_int = np.rint(original_paa).astype(int)
        
        # Combine all data
        augmented_data_original_scale = np.column_stack((augmented_xyz_int, pen_status, original_paa_int[:len(augmented_data)]))
        
        all_augmented_data.append(augmented_data_original_scale)
        
        # Visualization of the augmented data
        augmented_on_paper = augmented_data_original_scale[augmented_data_original_scale[:, 3] == 1]
        augmented_in_air = augmented_data_original_scale[augmented_data_original_scale[:, 3] == 0]

        # Scatter plot for the augmented data
        axs[i].scatter(augmented_on_paper[:, 1], augmented_on_paper[:, 0], c='b', s=1, label='On Paper')
        axs[i].scatter(augmented_in_air[:, 1], augmented_in_air[:, 0], c='r', s=1, label='In Air')
        axs[i].set_title(f'Augmented Data {i + 1}')
        axs[i].set_xlabel('y')
        axs[i].set_ylabel('x')
        axs[i].set_aspect('equal')
        axs[i].invert_xaxis()  # Flip the horizontal axis

    return all_augmented_data  # Return the list of augmented datasets after scaling back

def visualize_augmented_data_from_directory(directory):
    augmented_files = [f for f in os.listdir(directory) if f.startswith('augmented_') and f.endswith('.svc')]
    num_files = len(augmented_files)
    if num_files == 0:
        print("No augmented data files found in the directory.")
        return
    
    fig, axs = plt.subplots(1, num_files, figsize=(6 * num_files, 6), constrained_layout=True)
    if num_files == 1:
        axs = [axs]
    
    for i, filename in enumerate(augmented_files):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        df.columns = ['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']
        
        on_paper = df[df['pen_status'] == 1]
        in_air = df[df['pen_status'] == 0]

        axs[i].scatter(on_paper['y'], on_paper['x'], c='b', s=1, alpha=0.7, label='On Paper')
        axs[i].scatter(in_air['y'], in_air['x'], c='r', s=1, alpha=0.7, label='In Air')
        axs[i].set_title(f'Augmented Data {i + 1}')
        axs[i].set_xlabel('y')
        axs[i].set_ylabel('x')
        axs[i].invert_xaxis()
        axs[i].set_aspect('equal')
        axs[i].legend()
    
    plt.show()

def get_unique_filename(directory, filename):
    base, extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base}({counter}){extension}"
        counter += 1
    return filename

# 9. Download augmented data function
def download_augmented_data_with_modified_timestamp(augmented_datasets, scalers, original_data_frames, original_filenames, directory1='augmented_data'):
    global all_augmented_data  # Access the global list

    if not os.path.exists(directory1):
        os.makedirs(directory1)

    for i, (augmented_data, scaler, original_df, original_filename) in enumerate(zip(augmented_datasets, scalers, original_data_frames, original_filenames)):
        augmented_xyz = scaler.inverse_transform(augmented_data[:, :3])
        augmented_xyz_int = np.rint(augmented_xyz).astype(int)
        pen_status = augmented_data[:, 3].astype(int)
        original_paa = original_df[['pressure', 'azimuth', 'altitude']].values
        
        if len(augmented_data) > len(original_paa):
            original_paa = repeat_backwards(original_paa, len(augmented_data))
        
        original_paa_int = np.rint(original_paa).astype(int)
        
        new_timestamps = np.zeros(len(augmented_data), dtype=int)
        increment_sequence = [7, 8]
        current_time = 0
        for idx in range(len(augmented_data)):
            new_timestamps[idx] = current_time
            current_time += increment_sequence[idx % 2]

        augmented_xyz_int[:, 2] = new_timestamps

        augmented_data_original_scale = np.column_stack((
            augmented_xyz_int,
            pen_status,
            original_paa_int[:len(augmented_data)]
        ))

        # Save augmented data as .svc files
        augmented_filename = f"synthetic_{original_filename}"
        augmented_filename = get_unique_filename(directory1, augmented_filename)
        augmented_file_path = os.path.join(directory1, augmented_filename)

        np.savetxt(augmented_file_path, augmented_data_original_scale, fmt='%d', delimiter=' ')

        # Store augmented data
        all_augmented_data.append(augmented_data_original_scale)

        print(f"Augmented data saved to {augmented_file_path}")
        print(f"Shape of augmented data for {original_filename}: {augmented_data_original_scale.shape}")

    # After saving all augmented data, zip the files
    zip_file_path = os.path.join(directory1, 'augmented_data.zip')
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(directory1):
            for filename in filenames:
                if filename != 'augmented_data.zip':  # Avoid adding the zip file itself
                    file_path = os.path.join(foldername, filename)
                    zipf.write(file_path, arcname=filename)  # Add files to the zip

    print(f"Zipped augmented data saved at {zip_file_path}")

    # Return the path to the zip file
    return zip_file_path

# Nested augmentation function
def nested_augmentation(num_augmentations, num_files_to_use, data_frames, scalers, input_filenames, original_data_frames, model_path, avg_data_points, processed_data):
    print(f"Inside nested_augmentation: processed_data type={type(processed_data)}, value={processed_data}")
    vae_pretrained = load_pretrained_vae(model_path)
    if vae_pretrained is None:
        print("Error: Pretrained VAE model could not be loaded. Augmentation process halted.")
        return
    print("Pretrained VAE model loaded.")

    # Use existing data for the first iteration
    # global scalers, input_filenames, original_data_frames

    # Check processed_data before passing it
    if isinstance(processed_data, (list, np.ndarray)):
        num_files_to_use = len(processed_data)
    else:
        raise ValueError(f"processed_data is not iterable, got: {type(processed_data)}")

    for iteration in range(num_augmentations):
        print(f"Starting augmentation iteration {iteration + 1}")
        
        if iteration > 0:
            # Update the data for subsequent iterations
            directory = 'augmented_data_nested'
            data_frames, processed_data, scalers, avg_data_points, input_filenames, original_data_frames = upload_and_process_files(directory, num_files_to_use)
        print(f"processed_data after processing in iteration {iteration + 1}: type={type(processed_data)}, value={processed_data}")
        augmented_datasets = generate_augmented_datasets(vae_pretrained, processed_data, data_frames, num_augmentations, avg_data_points,
                                                     base_latent_variability, latent_variability_range)
        
        # Clear augmented_data_nested directory
        if os.path.exists('augmented_data_nested'):
            shutil.rmtree('augmented_data_nested')
        os.makedirs('augmented_data_nested')
        
        download_augmented_data_with_modified_timestamp(augmented_datasets, scalers, original_data_frames, input_filenames)
        
        print(f"Completed augmentation iteration {iteration + 1}")
    
    # Clear the augmented_data_nested directory after the last iteration
    if os.path.exists('augmented_data_nested'):
        shutil.rmtree('augmented_data_nested')
        print("Cleared augmented_data_nested directory after the final iteration.")
    
    print("Nested augmentation process completed.")
    # visualize_augmented_data_from_directory('augmented_data')

# 10. Repeat backwards function
def repeat_backwards(original_paa, augmented_length):
    repeat_count = augmented_length - len(original_paa)
    if repeat_count <= 0:
        return original_paa
    backwards_rows = np.empty((0, original_paa.shape[1]))
    for i in range(repeat_count):
        row_to_repeat = original_paa[-(i % len(original_paa) + 1)]
        backwards_rows = np.vstack((backwards_rows, row_to_repeat))
    return np.vstack((original_paa, backwards_rows))

# 11. Calculate NRMSE
def calculate_nrmse(original, predicted): 
    """Calculate the Normalized Root Mean Squared Error (NRMSE)."""
    if original.shape != predicted.shape:
        raise ValueError("The shapes of the original and predicted datasets must match.")
    
    mse = np.mean((original - predicted) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(original) - np.min(original))
    
    return nrmse

def calculate_nrmse_for_augmented_data(original_data_frames, augmented_data_list):
    """Calculate NRMSE for a list of original and augmented datasets."""
    nrmse_values = []

    for i, (original_df, augmented) in enumerate(zip(original_data_frames, augmented_data_list)):
        original_array = original_df[['x', 'y', 'timestamp', 'pen_status']].values
        augmented_array = augmented[:, :4]  # Assuming first 4 columns match original data structure

        nrmse = calculate_nrmse(original_array, augmented_array)
        nrmse_values.append(nrmse)
        print(f"NRMSE for dataset {i+1}: {nrmse:.4f}")

    # Calculate average NRMSE
    average_nrmse = np.mean(nrmse_values)
    print(f"Average NRMSE: {average_nrmse:.4f}")
    
    return nrmse_values, average_nrmse

# 12. Create LSTM classifier
def create_lstm_classifier(input_shape):
    """Create and compile an LSTM model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))  # Adding dropout to introduce randomness
    model.add(tf.keras.layers.LSTM(50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(2))  # Predict x and y
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data_for_lstm(real_data, synthetic_data):
    """Prepare real and synthetic data for LSTM input."""
    n_features = min(real_data.shape[1], synthetic_data.shape[1])
    
    # Trim the features to match
    real_data_trimmed = real_data[:, :n_features]
    synthetic_data_trimmed = synthetic_data[:, :n_features]
    
    X = np.vstack((real_data_trimmed, synthetic_data_trimmed))
    y = np.concatenate((np.ones(len(real_data)), np.zeros(len(synthetic_data))))
    return X, y

# 13. Post-Hoc Discriminative Score Function
def post_hoc_discriminative_score(real_data, synthetic_data, n_splits=10):
    """Calculate the post-hoc discriminative score using K-Fold cross-validation."""
    X, y = prepare_data_for_lstm(real_data, synthetic_data)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reshape the data for LSTM input (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = create_lstm_classifier((1, X_train.shape[2]))
        model.fit(X_train, y_train, epochs=3, batch_size=256, verbose=1)

        y_pred = (model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return mean_accuracy, std_accuracy

# 14. Prepare data for LSTM
def prepare_data(df, time_steps=5):
    """Prepare the data for LSTM input by creating sequences of specified length."""
    data = df[['x', 'y']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences of length `time_steps`
    X, y = [], []
    for i in range(len(data_scaled) - time_steps):
        X.append(data_scaled[i:i+time_steps])
        y.append(data_scaled[i+time_steps])
    
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
    # Predict and inverse transform
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)
    
    # Compute MAPE for each test sample
    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
    print(f"\nMAPE: {mape * 100:.2f}%")
    
    # Interpretation of MAPE
    if mape < 0.1:
        interpretation = "Excellent prediction"
    elif mape < 0.2:
        interpretation = "Good prediction"
    elif mape < 0.5:
        interpretation = "Fair prediction"
    else:
        interpretation = "Poor prediction"
    
    print(f"Interpretation: {interpretation}")
    
    return mape

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs, desc="Training Progress")

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()

# 15. K-Fold Cross-Validation for LSTM
def k_fold_cross_validation(X, y, scaler, n_splits=10):
    """Perform K-Fold cross-validation on the LSTM model and return mean and std of MAPE."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(1000))

    mape_values = []
    for train_index, test_index in kf.split(X):
        # Split data into training and testing sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create and train the model for each fold
        model = create_lstm_classifier((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=5, batch_size=512, verbose=0, callbacks=[CustomCallback()])
        
        # Evaluate the model and store MAPE
        mape = evaluate_model(model, X_test, y_test, scaler)
        mape_values.append(mape)

    # Calculate Mean and Standard Deviation of MAPE
    mean_mape = np.mean(mape_values)
    std_mape = np.std(mape_values)

    print(f"\nMean MAPE: {mean_mape * 100:.2f}%")
    print(f"Standard Deviation of MAPE: {std_mape * 100:.2f}%")

    return mean_mape, std_mape

# 16. Save model function
def save_model(vae, save_dir):
    """Save the trained VAE model to the specified directory."""
    model_save_path = os.path.join(save_dir, "final_vae_model.h5")
    vae.save(model_save_path)
    print(f"VAE model saved at {model_save_path}.")

# 17. Plotting function
def plot_training_history(generator_loss_history, reconstruction_loss_history, kl_loss_history, nrmse_history):
    """Plot the training loss history and NRMSE."""
    
    fig_loss = plt.figure(figsize=(10, 5))
    plt.plot(generator_loss_history, label='Generator Loss')
    plt.plot(reconstruction_loss_history, label='Reconstruction Loss')
    plt.plot(kl_loss_history, label='KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    fig_nrmse = plt.figure(figsize=(10, 5))
    plt.plot(nrmse_history, label='NRMSE')
    plt.xlabel('Epoch')
    plt.ylabel('NRMSE')
    plt.title('Normalized Root Mean Squared Error Over Epochs')
    plt.legend()
    
    return fig_loss, fig_nrmse

# 18. Latent space visualization
def visualize_latent_space(model, data, perplexity=5, learning_rate=200, n_iter=250):
    """Visualize the latent space of the model using t-SNE."""
    # Encode data into the latent space
    latent_means, _ = model.encode(tf.convert_to_tensor(data, dtype=tf.float32))
    latent_means_np = latent_means.numpy()
    
    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    latent_2d = tsne.fit_transform(latent_means_np)
    
    # Create a color map for the latent points
    norm = plt.Normalize(vmin=np.min(latent_means_np), vmax=np.max(latent_means_np))
    cmap = plt.cm.cividis  # You can change the colormap to 'plasma', 'inferno', etc.
    colors = cmap(norm(latent_means_np).sum(axis=1))  # Coloring based on the sum of latent variables
    
    # Plot the 2D t-SNE result with the color map
    fig_latent = plt.figure(figsize=(16, 12))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, s=5, alpha=0.6)
    plt.colorbar(scatter)  # Add a color bar for the gradient
    plt.title('Latent Space Visualization using t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    return fig_latent

# Model parameters
latent_dim = 512
beta = 0.000001
learning_rate = 0.001

# Create instances of the models
vae = VAE(latent_dim, beta)
optimizer = tf.keras.optimizers.Adam(learning_rate)
lstm_discriminator = LSTMDiscriminator()

# Global latent variability settings
base_latent_variability = 100.0
latent_variability_range = (0.99, 1.01)
all_augmented_data = []