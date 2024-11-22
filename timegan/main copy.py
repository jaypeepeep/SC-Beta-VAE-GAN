import os
import numpy as np
import tensorflow as tf
import logging
import time
import tensorflow.keras.saving
import tensorflow.keras.utils
from tensorflow import keras


# Set up logging
logging.basicConfig(
    filename='process_svc_folder.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the loss function without the register_keras_serializable decorator
@keras.saving.register_keras_serializable()
def combined_loss(y_true, y_pred):
    """
    Combined loss function to maintain data distribution and temporal dynamics
    """
    # Mean Squared Error
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Temporal consistency loss
    temporal_true = y_true[:, 1:] - y_true[:, :-1]
    temporal_pred = y_pred[:, 1:] - y_pred[:, :-1]
    temporal_loss = tf.reduce_mean(tf.abs(temporal_true - temporal_pred))
    
    # Additional regularization
    variance_loss = tf.reduce_mean(tf.math.squared_difference(
        tf.math.reduce_mean(y_true, axis=0),
        tf.math.reduce_mean(y_pred, axis=0)
    ))
    
    # Weighted combination of losses
    return mse_loss + 0.1 * temporal_loss + 0.05 * variance_loss

class TimeGANAugmentor:
    def __init__(self, parameters=None):
        """Initialize TimeGAN with optional custom parameters."""
        self.parameters = parameters or {
            'hidden_dim': 8,
            'num_layers': 2,
            'iterations': 100,
            'batch_size': 128,
            'module': 'gru'
        }
        
        self.model = None
        self.embedder = None
        self.min_val = None
        self.max_val = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function(reduce_retracing=True)
    def train_step(self, X, Z):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            X = tf.cast(X, tf.float32)
            X_tilde = self.model([X, Z], training=True)
            loss = tf.reduce_mean(tf.square(X - X_tilde))
        
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss

    def _build_model(self, dim):
        """Build the TimeGAN model with multiple components."""
        hidden_dim = self.parameters['hidden_dim']
        num_layers = self.parameters['num_layers']

        X_input = tf.keras.layers.Input(shape=(None, dim), name='input_x')
        Z_input = tf.keras.layers.Input(shape=(None, dim), name='input_z')

        # Embedder Network
        embedder_input = X_input
        for i in range(num_layers):
            embedder_input = tf.keras.layers.GRU(
                hidden_dim, 
                return_sequences=True, 
                name=f'embedder_gru_{i}'
            )(embedder_input)
        
        H = tf.keras.layers.Dense(hidden_dim, activation='tanh', name='embedder_dense')(embedder_input)
        self.embedder = tf.keras.Model(X_input, H, name='embedder')

        # Recovery Network
        recovery_input = H
        for i in range(num_layers):
            recovery_input = tf.keras.layers.GRU(
                hidden_dim, 
                return_sequences=True, 
                name=f'recovery_gru_{i}'
            )(recovery_input)
        
        X_tilde = tf.keras.layers.Dense(dim, name='recovery_dense')(recovery_input)

        self.model = tf.keras.Model(
            inputs=[X_input, Z_input], 
            outputs=X_tilde
        )

        self.model.compile(
            optimizer=self.optimizer, 
            loss=combined_loss
        )

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

    def train(self, ori_data):
        """Train the TimeGAN model on input data."""
        ori_data, self.min_val, self.max_val = self._minmax_scale(ori_data)
        no, seq_len, dim = ori_data.shape

        self._build_model(dim)

        for it in range(self.parameters['iterations']):
            Z_mb = np.random.normal(
                loc=0, 
                scale=1, 
                size=[no, seq_len, dim]
            )

            loss = self.model.train_on_batch(
                [ori_data, Z_mb], 
                ori_data
            )
            
            if it % 100 == 0:
                print(f"Iteration {it}: Loss = {loss}")

    def save_model(self, save_dir='./timegan_model'):
        """Save the trained model."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaling parameters
        np.savez(
            os.path.join(save_dir, 'scaling_params.npz'), 
            min_val=self.min_val, 
            max_val=self.max_val
        )

        # Save model architecture as json
        model_json = self.model.to_json()
        with open(os.path.join(save_dir, 'model_architecture.json'), 'w') as f:
            f.write(model_json)

        # Save model weights
        self.model.save_weights(os.path.join(save_dir, 'model.weights.h5'))

        # Save embedder if it exists
        if self.embedder:
            embedder_json = self.embedder.to_json()
            with open(os.path.join(save_dir, 'embedder_architecture.json'), 'w') as f:
                f.write(embedder_json)
            self.embedder.save_weights(os.path.join(save_dir, 'embedder.weights.h5'))

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

        # Load model architecture from json
        with open(os.path.join(save_dir, 'model_architecture.json'), 'r') as f:
            model_json = f.read()
        self.model = tf.keras.models.model_from_json(model_json)
        
        # Load model weights
        self.model.load_weights(os.path.join(save_dir, 'model.weights.h5'))
        
        # Recompile the model
        self.model.compile(optimizer=self.optimizer, loss=combined_loss)

        # Load embedder if it exists
        try:
            with open(os.path.join(save_dir, 'embedder_architecture.json'), 'r') as f:
                embedder_json = f.read()
            self.embedder = tf.keras.models.model_from_json(embedder_json)
            self.embedder.load_weights(os.path.join(save_dir, 'embedder.weights.h5'))
        except Exception as e:
            print(f"Embedder model not found or could not be loaded: {e}")

    def generate_synthetic_data(self, ori_data):
        """Generate synthetic data using the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        no, seq_len, dim = ori_data.shape
        
        Z_mb = np.random.normal(
            loc=0, 
            scale=1, 
            size=[no, seq_len, dim]
        )

        # Use model's serving signature for prediction
        synthetic_data = self.model([ori_data, Z_mb], training=False)
        if isinstance(synthetic_data, tf.Tensor):
            synthetic_data = synthetic_data.numpy()

        synthetic_data = self._reverse_minmax_scaling(synthetic_data)
        synthetic_data = np.round(synthetic_data).astype(int)

        return synthetic_data

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
        input_folder = "./timegan/train"
        output_folder = "./timegan/output"
        model_save_dir = "./timegan/saved_model"

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        train_data = load_svc_files(input_folder)

        timegan = TimeGANAugmentor()
        timegan.train(train_data)

        timegan.save_model(model_save_dir)

        augmentation_folder = "./timegan/to_augment"
        augmented_folder = "./timegan/augmented"

        os.makedirs(augmented_folder, exist_ok=True)

        loaded_timegan = TimeGANAugmentor()
        loaded_timegan.load_model(model_save_dir)

        for file_name in os.listdir(augmentation_folder):
            if file_name.endswith('.svc'):
                file_path = os.path.join(augmentation_folder, file_name)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                seq_len = int(lines[0].strip())
                data = np.array([list(map(float, line.split())) for line in lines[1:]])
                data = data.reshape(-1, seq_len, data.shape[1])

                synthetic_data = loaded_timegan.generate_synthetic_data(data)

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