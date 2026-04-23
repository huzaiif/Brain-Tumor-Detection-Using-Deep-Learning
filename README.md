# NeuroSync AI: Brain Tumor Detection System

An enterprise-grade, AI-powered diagnostic interface designed to detect and classify brain tumors from MRI/CT scans. This system fuses a fine-tuned deep learning Convolutional Neural Network (CNN) with a premium web dashboard and real-time clinical insights powered by Google's Gemini Vision AI.

## 🧠 Deep Learning Architecture

The core classification engine is built using **Transfer Learning** on the **VGG16** architecture. The model was trained in a Jupyter Notebook environment (`brain_tumour.ipynb`) and exported for deployment.

### Model Details
- **Base Architecture**: VGG16 (pre-trained on ImageNet).
- **Input Shape**: `128x128x3` RGB images.
- **Fine-Tuning Strategy**: 
  - The majority of the VGG16 convolutional blocks were frozen to retain generalized feature extraction capabilities.
  - The final convolutional blocks (`layers[-4]` to `layers[-2]`) were unfrozen and fine-tuned specifically on brain MRI structural variances.
- **Classification Head**: Custom fully connected (Dense) layers with Dropout regularization to prevent overfitting.
- **Optimizer**: Adam.
- **Target Classes (4)**:
  1. Glioma
  2. Meningioma
  3. Pituitary Tumor
  4. No Tumor (Healthy)

### Training & Evaluation
The model training pipeline (detailed in the included `.ipynb`) involves robust image augmentation and evaluation techniques, outputting classification reports, confusion matrices, and ROC-AUC curves to ensure high diagnostic precision.

## 💻 System Features

1. **AI Classification**: Upload an MRI/CT scan to receive an instant classification and confidence index from the local VGG16 model.
2. **Gemini Clinical Insights**: The system passes the image and the CNN prediction to Google's **Gemini 2.5 Flash Vision Model** to generate a contextual, professional clinical summary.
3. **Premium Dashboard**: A responsive, 2-column input/output interface featuring advanced CSS glassmorphism, modern typography, and perfect spatial symmetry.

## 🚀 Installation & Setup

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
4. Run the Flask server:
   ```bash
   python main.py
   ```
5. Open `http://127.0.0.1:5000` in your browser.

## 📁 Repository Structure
- `main.py`: The Flask server routing and Gemini API integration logic.
- `templates/index.html`: The premium glassmorphism frontend.
- `brain_tumour.ipynb`: The Jupyter Notebook containing the data pipeline, VGG16 model definition, and training loops.
- `models/`: Directory housing the trained `model.h5` binary.
