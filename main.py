from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('models/model.h5')

# Class labels
class_labels = ['glioma', 'pituitary', 'meningioma', 'notumor']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

def generate_clinical_insight(image_path, result_text, confidence_score):
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return "Gemini API Key not found. Please add it to your .env file."
        
    try:
        client = genai.Client()
        img = Image.open(image_path)
        
        prompt = f"You are a highly advanced AI medical assistant analyzing a brain MRI/CT scan. Our local Convolutional Neural Network has classified this scan as '{result_text}' with a confidence score of {confidence_score}%. Please provide a short, professional, 2-3 sentence clinical insight about what this specific classification means, typical next steps, and a brief disclaimer to consult a human radiologist or specialist. Keep it concise."
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img, prompt]
        )
        return response.text
    except Exception as e:
        return f"Error generating AI insight: {str(e)}"

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)
            rounded_confidence = round(float(confidence)*100, 2)
            
            # Generate AI Insight from Gemini
            ai_insight = generate_clinical_insight(file_location, result, rounded_confidence)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=rounded_confidence, file_path=f'/uploads/{file.filename}', ai_insight=ai_insight)

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)