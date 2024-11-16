import os
import numpy as np
import tensorflow as tf
import logging
import argparse
import time

# Set up logging
logging.basicConfig(
    filename='timegan_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define TimeGAN function
def timegan(ori_data, parameters, mode='train', model_path='./timegan_model/timegan.ckpt'):
    """TimeGAN for generating synthetic time-series data."""
    tf.compat.v1.disable_eager_execution()

    # Data dimensions
    no, seq_len, dim = ori_data.shape

    # Reverse scaling
    def reverse_minmax_scaling(data, min_val, max_val):
        return data * (max_val + 1e-7) + min_val
    
    # Min-Max Scaler
    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    # Normalize data
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layers']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    gamma = 1

    # Input placeholders
    X = tf.compat.v1.placeholder(tf.float32, [None, None, dim], name="input_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, None, dim], name="input_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="input_t")

    # RNN cell type
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
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            rnn = tf.keras.layers.RNN(cells, return_sequences=True)
            return rnn(X)

    # Recovery
    def recovery(H, T):
        with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            rnn = tf.keras.layers.RNN(cells, return_sequences=True)
            return tf.keras.layers.Dense(dim)(rnn(H))

    # Generator
    def generator(Z, T):
        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            g_cell = tf.keras.layers.RNN(cells, return_sequences=True)
            g_outputs = g_cell(Z)
            return tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)(g_outputs)

    # Supervisor
    def supervisor(H, T):
        with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)]
            s_cell = tf.keras.layers.RNN(cells, return_sequences=True)
            s_outputs = s_cell(H)
            return tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)(s_outputs)

    # Discriminator
    def discriminator(H, T):
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            d_cell = tf.keras.layers.RNN(cells, return_sequences=True)
            d_outputs = d_cell(H)
            return tf.keras.layers.Dense(1, activation=None)(d_outputs)

    # Model architecture
    H = embedder(X, T)
    X_tilde = recovery(H, T)
    E_hat = generator(Z, T)
    H_hat_supervise = supervisor(H, T)
    H_hat = E_hat + H_hat_supervise
    X_hat = recovery(H_hat, T)

    # Losses
    e_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(X, X_tilde))

    # Optimizers
    embedder_optimizer = tf.compat.v1.train.AdamOptimizer().minimize(e_loss)

    # Session
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        if mode == 'train':
            sess.run(tf.compat.v1.global_variables_initializer())
            for it in range(iterations):
                Z_mb = np.random.uniform(0, 1, [no, seq_len, dim])
                T_mb = [seq_len] * no
                _, e_loss_val = sess.run([embedder_optimizer, e_loss], feed_dict={X: ori_data, Z: Z_mb, T: T_mb})
                if it % 20 == 0:
                    print(f"Iteration {it}: Embedder loss = {e_loss_val}")
            saver.save(sess, model_path)
            print("Model saved.")
        else:
            saver.restore(sess, model_path)
            print("Model loaded.")
        
        # Generate synthetic data
        T_mb = [seq_len] * no
        synthetic_data = sess.run(X_tilde, feed_dict={X: ori_data, T: T_mb})
    
    synthetic_data = reverse_minmax_scaling(synthetic_data, min_val, max_val)
    return np.round(synthetic_data).astype(int)

# Load SVC data
def load_svc(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    seq_len = int(lines[0].strip())
    data = np.array([list(map(float, line.split())) for line in lines[1:]])
    return data.reshape(-1, seq_len, data.shape[1])

# Process SVC folder
def process_svc_folder(input_folder, output_folder, parameters, mode='train'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".svc"):
            file_path = os.path.join(input_folder, file_name)
            ori_data = load_svc(file_path)
            synthetic_data = timegan(ori_data, parameters, mode)
            output_path = os.path.join(output_folder, f"synthetic_{file_name}")
            with open(output_path, 'w') as f:
                f.write(f"{synthetic_data.shape[1]}\n")
                for row in synthetic_data.reshape(-1, synthetic_data.shape[-1]):
                    f.write(" ".join(map(str, row)) + "\n")
    print("Processing completed.")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='./timegan/batch', help='Input folder path')
    parser.add_argument('--output_folder', type=str, default='./timegan/output', help='Output folder path')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train', help='Mode: train or generate')
    args = parser.parse_args()

    timegan_params = {
        'hidden_dim': 24,
        'num_layers': 3,
        'iterations': 200,
        'batch_size': 32,
        'module': 'gru'
    }

    process_svc_folder(args.input_folder, args.output_folder, timegan_params, args.mode)
