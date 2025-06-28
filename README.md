# FaceTruth - DeepFake Detection

A web application for detecting deepfake images using the MesoNet architecture.

## Overview

This project implements a deepfake detection system using the MesoNet deep learning architecture. The web interface allows users to upload images and receive instant feedback on whether the image is likely to be authentic or a deepfake.

## Features

- Modern, responsive UI with Tailwind CSS
- Real-time deepfake detection
- Visual result presentation with confidence scores
- Simple and intuitive user interface

## Screenshots

Here are some screenshots of the live demo:
![Screenshot (13)](https://github.com/user-attachments/assets/173433c4-0f46-4745-ac04-183c1a036e5d)

![Screenshot (14)](https://github.com/user-attachments/assets/3a7a8cc4-6e74-4dba-86d7-2768d28e97bf)

![Screenshot (15)](https://github.com/user-attachments/assets/1190fa02-0ff6-4fd3-8d12-c29cf6cee8b7)

![Screenshot (16)](https://github.com/user-attachments/assets/22ebe378-9855-4d0d-b345-c2ac4090c4da)

## How It Works

1. Upload an image through the web interface
2. The MesoNet model analyzes the image for signs of manipulation
3. Results are displayed with a confidence score and visualization
4. Get instant feedback on image authenticity

## Tech Stack

- Python 3.x
- Flask
- PyTorch
- Tailwind CSS
- Matplotlib

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/DeepFake_Detection_Using_MesoNet.git
cd DeepFake_Detection_Using_MesoNet
```

2. Create a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
```
pip install -r requirements.txt
```

5. Run the application:
```
python app.py
```

6. Open your browser and navigate to: `http://127.0.0.1:5000`

## Project Structure

- `app.py`: Flask web application
- `model.py`: MesoNet model implementation
- `templates/`: HTML templates
- `static/`: Static assets (CSS, JS, images)
- `mesonet_model.pth`: Pre-trained model weights

## Team

- Soham Chafale
- Aditya Kinikar
- Rajveer R.
- Rajeev P.

## License

MIT

