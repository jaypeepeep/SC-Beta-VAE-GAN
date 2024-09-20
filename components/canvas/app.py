from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Set the folder where files will be saved
UPLOAD_FOLDER = 'uploads'  # You can change this to any folder you want
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_svc', methods=['POST'])
def upload_svc():
    # Check if the POST request has the file part
    if 'svc_file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    svc_file = request.files['svc_file']
    
    if svc_file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Save the file in the UPLOAD_FOLDER
    svc_file_path = os.path.join(UPLOAD_FOLDER, svc_file.filename)
    svc_file.save(svc_file_path)
    
    return jsonify({'message': f'File saved at {svc_file_path}'}), 200

if __name__ == '__main__':
    app.run(debug=True)
