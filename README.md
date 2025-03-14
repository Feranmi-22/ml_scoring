# Loan Default Prediction API

This API analyzes audio responses from loan applicants to predict the likelihood of loan default using machine learning.

## Overview

The Loan Default Prediction API processes three audio responses from loan applicants:

1. How they plan to pay back the loan
2. Their loan history
3. The purpose of the loan

Audio features are extracted using advanced signal processing techniques and fed into a trained CNN model to predict whether an applicant is likely to default on their loan.

## Features

- Audio processing from URLs (supports both direct audio and video files)
- Automatic extraction of audio from video files using FFmpeg
- Feature extraction using librosa for advanced audio analysis
- Machine learning prediction using a pre-trained CNN model
- RESTful API built with FastAPI for easy integration

## Requirements

- Python 3.8+
- FFmpeg installed on the system
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ml-loan-scoring.git
cd ml-loan-scoring
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure FFmpeg is installed on your system:

   - For Ubuntu/Debian: `apt-get install ffmpeg`
   - For macOS: `brew install ffmpeg`
   - For Windows: Download from [FFmpeg official website](https://ffmpeg.org/download.html)

4. Update the `MODEL_PATH` in `app.py` to point to your trained model file.

## Usage

1. Start the API server:

```bash
python app.py
```

2. The API will be available at `http://localhost:8000`

3. Use the `/predict` endpoint with a POST request containing the URLs for the three audio responses:

```json
{
  "audio_how_to_pay_back": "https://example.com/response1.mp4",
  "audio_loan_history": "https://example.com/response2.wav",
  "audio_loan_purpose": "https://example.com/response3.wav"
}
```

4. The API will return a prediction:

```json
{
  "prediction": "paid" or "defaulted"
}
```

## API Endpoints

- `GET /`: Basic information about the API
- `POST /predict`: Submit audio URLs for loan default prediction

## Model Details

The API uses a CNN model trained on audio features to predict loan default probability. The model analyzes patterns in speech that correlate with loan repayment behavior.

Key features extracted from audio include:

- Spectral centroid
- Spectral bandwidth
- Spectral contrast
- Spectral rolloff
- Zero crossing rate
- RMS energy
- Pitch information
- MFCCs (Mel-frequency cepstral coefficients)

The model returns a binary classification determining default risk.
