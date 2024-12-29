## Project Overview
The Air Pollution Prediction App is a web-based application designed to predict air pollution levels using machine learning models. It provides users with an easy-to-use interface to input relevant data and receive predictions about air quality. The app is built with Streamlit and leverages pre-trained models to ensure accurate and efficient predictions.

This project aims to raise awareness about air pollution and provide actionable insights for mitigating its impact

## Project Structure

Project/
├── workspace/
│   └── notebook.ipynb          # Jupyter Notebook for analysis and experimentation
├── src/
│   ├── app.py                  # Streamlit web application
│   ├── preprocessing.py        # Data preprocessing script
│   ├── model.py                # Model training script
│   ├── trained_model.pkl       # Saved pre-trained model
│   ├── encoder.pkl             # Saved encoder for categorical data
│   ├── feature_names.pkl       # Saved feature names for the model
│   ├── scaler.pkl              # Saved scaler for data normalization
├── README.md 
├── dataset/
│   └── updated_pollution_dataset.csv        # Dataset for training                
└── requirements.txt            # Project dependencies


## Setup and Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run src/app.py
```


## Model Details
- Algorithm: Logistic regression 
- Features: CO2 ,So2,nearst industrial areas , etc.
- Performance Metrics available in model training logs




## The app can be deployed on platforms like:
Streamlit Cloud



## Contributions
Feel free to fork and improve the project!

