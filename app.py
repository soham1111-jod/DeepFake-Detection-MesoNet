from flask import Flask, render_template, request
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from werkzeug.utils import secure_filename
from model import MesoNet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set device to CPU
device = torch.device("cpu")

# Initialize the model
model = MesoNet().to(device)

# Load the pretrained model weights onto the CPU
model.load_state_dict(torch.load("mesonet_model.pth", map_location=torch.device('cpu')))
model.eval()

def predict(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        probs = outputs[0].cpu().detach().numpy()  # Get probabilities
        return probs

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html')

        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html')
        if file:
            # Save the uploaded image to the static images folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get the prediction for the image
            prob = predict(file_path)
            prob = prob[0]
            
            # Calculate the probability percentages
            prob_real = prob * 100
            prob_fake = (1 - prob) * 100
            
            # Determine if the image is real or fake
            if prob > 0.5:
                prediction_label = 'Real'
            else:
                prediction_label = 'Fake'

            # Create a bar graph for the prediction
            labels = ['Real', 'Fake']
            plt.figure(figsize=(6, 4), facecolor='#1F2937')
            ax = plt.gca()
            ax.set_facecolor('#1F2937')
            plt.bar(labels, [prob_real, prob_fake], color=['#10B981', '#EF4444'], width=0.5)
            plt.ylabel('Confidence (%)', color='white')
            plt.title('Detection Results', color='white', fontsize=14, fontweight='bold')
            plt.ylim(0, 100)
            plt.yticks([0, 20, 40, 60, 80, 100], color='white')
            plt.xticks(color='white')
            
            # Add percentage labels on the bars
            for i, v in enumerate([prob_real, prob_fake]):
                plt.text(i, v + 2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
                
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Save the graph to a buffer and convert it to a base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graph_string = base64.b64encode(buffer.getvalue()).decode()
            
            # Render the result with the graph
            return render_template('predict.html', 
                                   graph=graph_string, 
                                   prediction=prediction_label, 
                                   filename=filename,
                                   prob_real=f"{prob_real:.1f}",
                                   prob_fake=f"{prob_fake:.1f}")

    return render_template('predict.html')

@app.route('/ping')
def ping():
    return {'status': 'ok'}, 200

if __name__ == '__main__':
    app.run(debug=True)
