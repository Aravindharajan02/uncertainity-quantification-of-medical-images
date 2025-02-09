# uncertainity-quantification-of-medical-images
# Brain Tumor MRI Analysis System 🧠

A deep learning-based web application for analyzing brain MRI scans to detect tumors. Built with PyTorch and Streamlit.

## Features

- 🔍 Real-time MRI scan analysis
- 📊 Uncertainty estimation using Monte Carlo dropout
- 🎯 Adjustable confidence thresholds
- 📈 Interactive visualization of results
- 🖼️ Advanced image preprocessing
- 💻 User-friendly web interface

## Demo

![Demo Screenshot](assets/demo_screenshot.png)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aravindharajan02/uncertainity-quantification-of-medical-images
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage


1. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

## Project Structure

```
brain-tumor-mri-analysis/
├── app.py              # Streamlit web application
├── train.py           # Model training script
├── models/            # Directory for saved models
├── requirements.txt   # Project dependencies
├── .gitignore        # Git ignore file
└── README.md         # Project documentation
```

## Model Architecture

- Base model: ResNet18 with transfer learning
- Custom modifications for tumor detection
- Monte Carlo dropout for uncertainty estimation
- Binary classification (tumor/no tumor)

## Technologies Used

- PyTorch
- Streamlit
- OpenCV
- Plotly
- NumPy
- PIL

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: Brain Tumor MRI Dataset from Kaggle
- ResNet architecture by Microsoft Research
- Streamlit for the amazing web framework

## Contact

Aravindharajan_S_S saravanan.aravindh02@gmail.com
