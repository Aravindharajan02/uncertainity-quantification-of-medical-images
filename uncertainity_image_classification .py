import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import cv2
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
import matplotlib.pyplot as plt

class EnhancedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.base_model(x)

def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedCNN().to(device)
    if os.path.exists('models/best_model.pth'):
        model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    else:
        st.warning("⚠️ Model file not found. Using initialized model.")
    model.eval()
    return model

def predict_with_uncertainty(model, image, num_samples=50):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image = transform(image=np.array(image))['image'].unsqueeze(0)
    predictions = []
    model.train()
    for _ in range(num_samples):
        with torch.no_grad():
            pred = torch.softmax(model(image), dim=1)
            predictions.append(pred.cpu().numpy())
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)

def main():
    st.set_page_config(page_title="Brain MRI Analyzer", page_icon="🧠", layout="wide")
    st.title("🧠 Brain Tumor MRI Analysis")
    model = load_model()
    with st.sidebar:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
        mc_samples = st.slider("Monte Carlo Samples", 10, 100, 50)
        show_preprocessing = st.checkbox("Show Preprocessing Steps", value=False)
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')
            processed_image = preprocess_image(original_image)
            if show_preprocessing:
                col_orig, col_proc = st.columns(2)
                with col_orig: st.image(original_image, caption="Original")
                with col_proc: st.image(processed_image, caption="Processed")
            else:
                st.image(processed_image, caption="MRI Scan")
            if st.button("Analyze"):
                with st.spinner("Analyzing image..."):
                    mean_pred, std_pred = predict_with_uncertainty(model, processed_image, mc_samples)
                    with col2:
                        pred_class = np.argmax(mean_pred)
                        confidence = mean_pred[0][pred_class]
                        uncertainty = std_pred[0][pred_class]
                        result_color = "green" if confidence >= confidence_threshold else "orange"
                        st.markdown(f"""
                            <div style='background-color: {result_color}; padding: 20px; border-radius: 10px; color: white;'>
                                <h3>{'🔴 Tumor Detected' if pred_class == 1 else '✅ No Tumor Detected'}</h3>
                                <p>Confidence: {confidence:.2%}</p>
                                <p>Uncertainty: ±{uncertainty:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        if confidence < confidence_threshold:
                            st.warning("⚠️ Low confidence prediction. Please consult a medical professional.")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['No Tumor', 'Tumor'],
                            y=mean_pred[0],
                            error_y=dict(type='data', array=std_pred[0]),
                            marker_color=['blue', 'red']
                        ))
                        fig.update_layout(title='Prediction Probabilities with Uncertainty', yaxis_title='Probability', yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)

                        # Display confusion matrix
                        y_true = [1 if pred_class == 1 else 0]
                        y_pred = [1 if confidence >= confidence_threshold else 0]
                        cm = confusion_matrix(y_true, y_pred)
                        fig_cm, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig_cm)

if __name__ == "__main__":
    main()
