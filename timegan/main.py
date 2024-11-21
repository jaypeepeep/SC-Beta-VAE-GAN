import os
import numpy as np
import tensorflow as tf
import logging
import time

# Set up logging
logging.basicConfig(
    filename='process_svc_folder.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TimeGANAugmentor:
    def __init__(self, parameters=None):
        """Initialize TimeGAN with optional custom parameters."""
        self.parameters = parameters or {
            'hidden_dim': 24,
            'num_layers': 3,
            'iterations': 500,
            'batch_size': 32,
            'module': 'gru'
        }
        
        # Model components
        self.model = None
        self.embedder = None
        
        # Scaling parameters
        self.min_val = None
        self.max_val = None

    @tf.function(reduce_retracing=True)
    def train_step(self, X, Z):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            # Ensure consistent data types
            X = tf.cast(X, tf.float32)
            X_tilde = self.model([X, Z], training=True)
            
            # Compute Mean Squared Error
            loss = tf.reduce_mean(tf.square(X - X_tilde))
        
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss



    def _build_model(self, dim):
        """Build the TimeGAN model using functional API."""
        hidden_dim = self.parameters['hidden_dim']
        num_layers = self.parameters['num_layers']

        # Input layers
        X_input = tf.keras.layers.Input(shape=(None, dim), name='input_x')
        Z_input = tf.keras.layers.Input(shape=(None, dim), name='input_z')

        # Embedder network
        embedder_input = X_input
        for _ in range(num_layers):
            embedder_input = tf.keras.layers.GRU(
                hidden_dim, 
                return_sequences=True, 
                name=f'embedder_gru_{_}'
            )(embedder_input)
        
        H = tf.keras.layers.Dense(hidden_dim, name='embedder_dense')(embedder_input)
        self.embedder = tf.keras.Model(X_input, H, name='embedder')

        # Recovery network
        recovery_input = H
        for _ in range(num_layers):
            recovery_input = tf.keras.layers.GRU(
                hidden_dim, 
                return_sequences=True, 
                name=f'recovery_gru_{_}'
            )(recovery_input)
        
        X_tilde = tf.keras.layers.Dense(dim, name='recovery_dense')(recovery_input)

        # Complete model
        self.model = tf.keras.Model(
            inputs=[X_input, Z_input], 
            outputs=X_tilde
        )

        # Compile model
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, ori_data):
        """Train the TimeGAN model on input data."""
        # Normalize data
        ori_data, self.min_val, self.max_val = self._minmax_scale(ori_data)
        no, seq_len, dim = ori_data.shape

        # Build model
        self._build_model(dim)

        # Training loop
        for it in range(self.parameters['iterations']):
            # Generate random noise
            Z_mb = np.random.uniform(0, 1, [no, seq_len, dim])

            # Train using custom train step
            loss = self.train_step(ori_data, Z_mb)
            print(f"Iteration {it}: Embedder loss = {loss.numpy()}")

    def save_model(self, save_dir='./timegan_model'):
        """Save the trained model."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaling parameters
        np.savez(
            os.path.join(save_dir, 'scaling_params.npz'), 
            min_val=self.min_val, 
            max_val=self.max_val
        )

        # Save entire model with .keras extension
        model_path = os.path.join(save_dir, 'timegan_model.keras')
        self.model.save(model_path)
        
        # Optionally save embedder separately
        if self.embedder:
            embedder_path = os.path.join(save_dir, 'embedder_model.keras')
            self.embedder.save(embedder_path)

    def load_model(self, save_dir='./timegan_model'):
        """Load a pre-trained model."""
        # Load scaling parameters
        scaling_file = os.path.join(save_dir, 'scaling_params.npz')
        if os.path.exists(scaling_file):
            scaling_data = np.load(scaling_file)
            self.min_val = scaling_data['min_val']
            self.max_val = scaling_data['max_val']
        else:
            raise FileNotFoundError("Scaling parameters not found")

        # Load entire model
        model_path = os.path.join(save_dir, 'timegan_model.keras')
        self.model = tf.keras.models.load_model(model_path)

        # Optionally load embedder
        try:
            embedder_path = os.path.join(save_dir, 'embedder_model.keras')
            self.embedder = tf.keras.models.load_model(embedder_path)
        except Exception:
            print("Embedder model not found or could not be loaded")

    def generate_synthetic_data(self, ori_data):
        """Generate synthetic data using the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        no, seq_len, dim = ori_data.shape
        Z_mb = np.random.uniform(0, 1, [no, seq_len, dim])

        # Generate synthetic data
        synthetic_data = self.model.predict([ori_data, Z_mb])

        # Reverse normalization
        synthetic_data = self._reverse_minmax_scaling(synthetic_data)
        synthetic_data = np.round(synthetic_data).astype(int)

        return synthetic_data

    def _minmax_scale(self, data):
        """Min-Max scaling of input data."""
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    def _reverse_minmax_scaling(self, data):
        """Reverse Min-Max scaling."""
        return data * (self.max_val + 1e-7) + self.min_val

def load_svc_files(input_folder):
    """Load all SVC files from the input folder."""
    svc_files = [f for f in os.listdir(input_folder) if f.endswith('.svc')]
    all_data = []

    for file_name in svc_files:
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        seq_len = int(lines[0].strip())
        data = np.array([list(map(float, line.split())) for line in lines[1:]])
        data = data.reshape(-1, seq_len, data.shape[1])
        all_data.append(data)

    return np.concatenate(all_data)

def main():
    try:
        # Input and output folders
        input_folder = "./timegan/train"
        output_folder = "./timegan/output"
        model_save_dir = "./timegan/saved_model"

        # Create output and model directories
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        # Load training data
        train_data = load_svc_files(input_folder)

        # Initialize and train TimeGAN
        timegan = TimeGANAugmentor()
        timegan.train(train_data)

        # Save the trained model
        timegan.save_model(model_save_dir)

        # Augmentation phase 
        augmentation_folder = "./timegan/to_augment"
        augmented_folder = "./timegan/augmented"

        os.makedirs(augmented_folder, exist_ok=True)

        # Load the saved model
        loaded_timegan = TimeGANAugmentor()
        loaded_timegan.load_model(model_save_dir)

        # Process files for augmentation
        for file_name in os.listdir(augmentation_folder):
            if file_name.endswith('.svc'):
                file_path = os.path.join(augmentation_folder, file_name)
                
                # Load data to augment
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                seq_len = int(lines[0].strip())
                data = np.array([list(map(float, line.split())) for line in lines[1:]])
                data = data.reshape(-1, seq_len, data.shape[1])

                # Generate synthetic data
                synthetic_data = loaded_timegan.generate_synthetic_data(data)

                # Save synthetic data
                output_path = os.path.join(augmented_folder, f"synthetic_{file_name}")
                with open(output_path, 'w') as f:
                    f.write(f"{synthetic_data.shape[1]}\n")
                    for row in synthetic_data.reshape(-1, synthetic_data.shape[-1]):
                        f.write(" ".join(map(str, row)) + "\n")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()