import os
import numpy as np
import tensorflow as tf
import logging
import time

directory_name = "timegan"

# Ensure the directory exists
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# Set up logging inside the 'timegan' directory
log_file_path = os.path.join(directory_name, 'process_svc_folder.txt')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Define TimeGAN function
def timegan(ori_data, parameters):
    """TimeGAN with optimized parameters for faster training."""
    tf.compat.v1.disable_eager_execution()
    
    # Basic Parameters
    no, seq_len, dim = ori_data.shape
    
    # Normalize data
    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val
    
    def reverse_minmax_scaling(data, min_val, max_val):
        return data * (max_val + 1e-7) + min_val
    
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    
    # Optimized Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layers']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    
    # Reduce network complexity while maintaining quality
    gamma = 0.5  # Reduced from 1.0
    z_dim = dim
    # Input placeholders
    X = tf.compat.v1.placeholder(tf.float32, [None, None, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, None, dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

    # Define RNN cell type
    def rnn_cell(module_name, hidden_dim):
        if module_name == 'gru':
            return tf.keras.layers.GRUCell(hidden_dim)
        elif module_name == 'lstm':
            return tf.keras.layers.LSTMCell(hidden_dim)
        else:
            return tf.keras.layers.SimpleRNNCell(hidden_dim)

    # Embedder
    def embedder(X, T):
        with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [tf.keras.layers.GRUCell(hidden_dim) for _ in range(num_layers)]
            rnn = tf.keras.layers.RNN(cells, return_sequences=True)
            H = rnn(X)
        return H

    # Recovery
    def recovery(H, T):
        with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [tf.keras.layers.GRUCell(hidden_dim) for _ in range(num_layers)]
            rnn = tf.keras.layers.RNN(cells, return_sequences=True)
            H_output = rnn(H)
            X_tilde = tf.keras.layers.Dense(dim)(H_output)  # Ensure output matches input feature dimension
        return X_tilde

    # Generator
    def generator(Z, T):
        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            g_cell = tf.keras.layers.RNN(cells, return_sequences=True)
            g_outputs = g_cell(Z)
            E = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)(g_outputs)
        return E

    # Supervisor
    def supervisor(H, T):
        with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)]
            s_cell = tf.keras.layers.RNN(cells, return_sequences=True)
            s_outputs = s_cell(H)
            S = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)(s_outputs)
        return S

    # Discriminator
    def discriminator(H, T):
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            d_cell = tf.keras.layers.RNN(cells, return_sequences=True)
            d_outputs = d_cell(H)
            Y_hat = tf.keras.layers.Dense(1, activation=None)(d_outputs)
        return Y_hat

    # Build the TimeGAN model
    H = embedder(X, T)
    X_tilde = recovery(H, T)
    E_hat = generator(Z, T)
    H_hat_supervise = supervisor(H, T)
    H_hat = E_hat + H_hat_supervise
    X_hat = recovery(H_hat, T)
    Y_fake = discriminator(H_hat, T)
    Y_real = discriminator(H, T)
    Y_fake_e = discriminator(E_hat, T)

    # Loss functions
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_real, labels=tf.ones_like(Y_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake, labels=tf.zeros_like(Y_fake)))
    d_loss_fake_e = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake_e, labels=tf.zeros_like(Y_fake_e)))
    d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e

    g_loss_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake, labels=tf.ones_like(Y_fake)))
    g_loss_s = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(H_hat_supervise[:, :-1], H[:, 1:]))
    g_loss_v = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake_e, labels=tf.ones_like(Y_fake_e)))
    g_loss = g_loss_u + gamma * g_loss_s + 0.1 * g_loss_v

    e_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(X, X_tilde))

    # Optimizers using tf.keras.optimizers.Adam
    optimizer = tf.compat.v1.train.AdamOptimizer()

    # Training Loop
    train_embedder = optimizer.minimize(e_loss)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Use mini-batches for faster training
        num_batches = no // batch_size
        
        for it in range(iterations):
            start_time = time.time()  # Record start time
            # Process in mini-batches
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = start_idx + batch_size
                
                X_mb = ori_data[start_idx:end_idx]
                Z_mb = np.random.uniform(0, 1, [batch_size, seq_len, dim])
                T_mb = [seq_len] * batch_size
                
                # Train embedder
                _, e_loss_val = sess.run(
                    [train_embedder, e_loss],
                    feed_dict={X: X_mb, Z: Z_mb, T: T_mb}
                )
            
            # Record the time for this iteration
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Print progress less frequently
            if it % 20 == 0:
                print(f"Iteration {it}/{iterations}: Embedder loss = {e_loss_val}, Time taken = {elapsed_time:.2f} seconds")
        
        # Generate synthetic data
        synthetic_data = sess.run(X_tilde, feed_dict={X: ori_data, T: [seq_len] * no})
    
    # Reverse normalization and round to integer format
    synthetic_data = reverse_minmax_scaling(synthetic_data, min_val, max_val)
    synthetic_data = np.round(synthetic_data).astype(int)
    
    return synthetic_data

# Function to load and process SVC files
def load_svc(file_path, fixed_seq_len=64):
    """
    Load SVC file with fixed sequence length.
    
    Args:
        file_path: Path to the SVC file
        fixed_seq_len: Fixed sequence length to use for all sequences
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the first line (original sequence length) and convert data
    data = np.array([list(map(float, line.split())) for line in lines[1:]])
    
    # Calculate how many complete sequences we can make
    total_rows = len(data)
    num_complete_sequences = total_rows // fixed_seq_len
    
    if num_complete_sequences == 0:
        raise ValueError(f"File {file_path} has fewer rows ({total_rows}) than fixed_seq_len ({fixed_seq_len})")
    
    # Only keep complete sequences
    used_rows = num_complete_sequences * fixed_seq_len
    data = data[:used_rows]
    
    dim = data.shape[1]
    return data.reshape(-1, fixed_seq_len, dim)  # Reshape to (batch, fixed_seq_len, dim)

def process_svc_folder(input_folder, output_folder, parameters):
    """Process SVC files sequentially."""
    start_time = time.time()
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of SVC files
    svc_files = [f for f in os.listdir(input_folder) if f.endswith('.svc')]
    
    if not svc_files:
        logging.warning("No SVC files found in the input folder")
        return
    
    logging.info(f"Starting sequential processing of {len(svc_files)} files")
    
    successful_files = []
    failed_files = []
    total_processing_time = 0
    
    # Process each file sequentially
    for file_name in svc_files:
        try:
            file_start_time = time.time()
            file_path = os.path.join(input_folder, file_name)
            
            logging.info(f"Starting processing of file: {file_name}")
            
            # Load and process the data
            ori_data = load_svc(file_path, fixed_seq_len=32)
            
            # Generate synthetic data
            synthetic_data = timegan(ori_data, parameters)
            
            # Save synthetic data
            # Save synthetic data
            output_path = os.path.join(output_folder, f"synthetic_{file_name}")
            with open(output_path, 'w') as f:
                # Directly write the data without sequence length
                for row in synthetic_data.reshape(-1, synthetic_data.shape[-1]):
                    f.write(" ".join(map(str, row)) + "\n")
            
            file_end_time = time.time()
            processing_time = file_end_time - file_start_time
            total_processing_time += processing_time
            
            successful_files.append(file_name)
            logging.info(f"Completed processing file {file_name}. Time taken: {processing_time:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {str(e)}")
            failed_files.append((file_name, str(e)))
    
    # Log summary
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info(f"\nProcessing Summary:")
    logging.info(f"Total files processed: {len(svc_files)}")
    logging.info(f"Successfully processed: {len(successful_files)}")
    logging.info(f"Failed to process: {len(failed_files)}")
    logging.info(f"Total time taken: {total_time:.2f} seconds")
    logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
    
    if failed_files:
        logging.error("\nFailed files:")
        for file_name, error in failed_files:
            logging.error(f"{file_name}: {error}")

# Parameters
timegan_params = {
    'hidden_dim': 128,
    'num_layers': 2,
    'iterations': 100,
    'batch_size': 32,
    'module': 'gru'
}

if __name__ == '__main__':
    input_folder = "./timegan/train"
    output_folder = "./timegan/output"
    
    process_svc_folder(input_folder, output_folder, timegan_params)