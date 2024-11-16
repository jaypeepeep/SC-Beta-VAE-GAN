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

# Define TimeGAN function
def timegan(ori_data, parameters):
    """TimeGAN for generating synthetic time-series data."""
    # Disable eager execution for compatibility with tf.compat.v1
    tf.compat.v1.disable_eager_execution()

    # Basic Parameters
    no, seq_len, dim = ori_data.shape

    # Reverse scaling
    def reverse_minmax_scaling(data, min_val, max_val):
        return data * (max_val + 1e-7) + min_val
    
    # Define Min-Max Normalizer
    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    # Normalize data
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layers']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    gamma = 1
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

        for it in range(iterations):
            Z_mb = np.random.uniform(0, 1, [no, seq_len, dim])
            T_mb = [seq_len] * no

            # Train embedder
            _, e_loss_val = sess.run([train_embedder, e_loss], feed_dict={X: ori_data, Z: Z_mb, T: T_mb})

            if it % 20 == 0:
                print(f"Iteration {it}: Embedder loss = {e_loss_val}")

        # Generate synthetic data
        synthetic_data = sess.run(X_tilde, feed_dict={X: ori_data, T: T_mb})

    # Reverse normalization and round to integer format
    synthetic_data = reverse_minmax_scaling(synthetic_data, min_val, max_val)
    synthetic_data = np.round(synthetic_data).astype(int)

    return synthetic_data


# Function to load and process SVC files
def load_svc(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    seq_len = int(lines[0].strip())
    
    if seq_len <= 0:
        raise ValueError(f"Invalid sequence length {seq_len} in file {file_path}")
    
    data = np.array([list(map(float, line.split())) for line in lines[1:]])
    
    # Ensure the data length matches the expected sequence length
    if data.shape[0] % seq_len != 0:
        raise ValueError(f"Data length {data.shape[0]} is not a multiple of sequence length {seq_len}")
    
    dim = data.shape[1]
    return data.reshape(-1, seq_len, dim)  # Reshape to (batch, seq_len, dim)


def process_svc_folder(input_folder, output_folder, parameters):
    start_time = time.time()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".svc"):
            file_path = os.path.join(input_folder, file_name)
            
            ori_data = load_svc(file_path)

            # Generate synthetic data
            synthetic_data = timegan(ori_data, parameters)

            # Save synthetic data
            output_path = os.path.join(output_folder, f"synthetic_{file_name}")
            with open(output_path, 'w') as f:
                f.write(f"{synthetic_data.shape[1]}\n")
                for row in synthetic_data.reshape(-1, synthetic_data.shape[-1]):
                    f.write(" ".join(map(str, row)) + "\n")


    end_time = time.time()
    logging.info(f"Completed processing folder. Total time: {end_time - start_time:.2f} seconds.")


# Parameters
timegan_params = {
    'hidden_dim': 24,
    'num_layers': 3,
    'iterations': 200,
    'batch_size': 32,
    'module': 'gru'
}

# Run the script
input_folder = "./timegan/batch"
output_folder = "./timegan/output"
process_svc_folder(input_folder, output_folder, timegan_params)
