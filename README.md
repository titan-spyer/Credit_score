# Credit Score Prediction

This project uses a Random Forest Classifier to predict credit scoring risk (High Risk vs Low Risk).

## Dataset
The dataset is downloaded automatically using `kagglehub` from [Give Me Some Credit](https://www.kaggle.com/datasets/lihxlhx/give-me-some-credit).

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To retrain the model, run:
```bash
python train_model.py
```
This will generate `credit_model.pkl` and `credit_scaler.pkl`.

### Running the App
To run the Streamlit inference app:
```bash
streamlit run app.py
```
