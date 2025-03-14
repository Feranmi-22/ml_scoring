import os
import requests
import numpy as np
import librosa
import uvicorn
import tempfile
import traceback
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default probability based on audio features",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
THRESHOLD = 0.615

MODEL_PATH = "D:\\Projects\\ML-Loan-Scoring\\ml-loan-scoring\\ml_scoring_pipeline\\loan_default_cnn_model_iteration_29.h5"


class AudioURLs(BaseModel):
    audio_how_to_pay_back: str
    audio_loan_history: str
    audio_loan_purpose: str


class PredictionResponse(BaseModel):
    prediction: str


@app.on_event("startup")
def load_model_and_threshold():
    global MODEL
    try:
        MODEL = load_model(MODEL_PATH)
        print(f"Model loaded successfully. Using threshold: {THRESHOLD}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


async def download_file(url: str, is_video: bool = False) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()

        extension = ".mp4" if is_video else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Download error: {error_details}")
        raise HTTPException(
            status_code=400, detail=f"Error downloading {url}: {str(e)}"
        )


async def extract_audio_from_video(video_path: str) -> str:
    try:
        print(f"Extracting audio from video: {video_path}")

        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            audio_path,
        ]

        process = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        os.unlink(video_path)

        print(f"Audio extracted successfully to: {audio_path}")
        return audio_path
    except subprocess.SubprocessError as e:
        error_details = traceback.format_exc()
        print(f"FFmpeg error: {error_details}")
        print(f"FFmpeg stderr: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        raise HTTPException(
            status_code=500, detail=f"Error extracting audio from video: {str(e)}"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Video processing error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


async def preprocess_audio(file_path, target_sr=16000, trim_db=20):
    try:
        print(f"Processing audio file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")

        if file_size == 0:
            raise ValueError("Audio file is empty (0 bytes)")

        audio, sr = librosa.load(file_path, sr=target_sr)
        print(f"Audio loaded successfully. Sample rate: {sr}, Length: {len(audio)}")

        os.unlink(file_path)

        trimmed_audio, _ = librosa.effects.trim(audio, top_db=trim_db)
        print(f"Audio trimmed successfully. New length: {len(trimmed_audio)}")

        return trimmed_audio, sr
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Audio processing error: {error_details}")
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")


def extract_features(audio, sr, n_mfcc=13):
    try:
        features = []
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        rms_energy = librosa.feature.rms(y=audio)
        pitches, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        features.extend(
            [
                np.mean(spectral_centroid),
                np.mean(spectral_bandwidth),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate),
                np.mean(rms_energy),
                pitch_mean,
            ]
        )
        features.extend(np.mean(spectral_contrast, axis=1))
        features.extend(np.mean(mfcc, axis=1))

        return features
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Feature extraction error: {error_details}")
        raise Exception(f"Error extracting features: {str(e)}")


def predict_default(features):
    try:
        if MODEL is None:
            load_model_and_threshold()

        features_reshaped = np.array(features).reshape(1, -1, 1)
        print(f"Features shape after reshaping: {features_reshaped.shape}")

        prediction = MODEL.predict(features_reshaped)
        default_probability = float(prediction[0][1])

        print(f"Default probability: {default_probability}, Threshold: {THRESHOLD}")
        predicted_class = "defaulted" if default_probability >= THRESHOLD else "paid"

        return {"prediction": predicted_class}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        raise Exception(f"Error making prediction: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(audio_urls: AudioURLs):
    try:
        print("Received prediction request with URLs:")
        for key, url in audio_urls.dict().items():
            print(f"- {key}: {url}")

        audio_files = {}

        for key, url in audio_urls.dict().items():
            if key == "audio_how_to_pay_back":
                print(f"Processing potential video URL for {key}: {url}")
                file_path = await download_file(url, is_video=True)
                print(f"Downloaded to {file_path}")

                try:
                    audio_path = await extract_audio_from_video(file_path)
                    print(f"Extracted audio to {audio_path}")

                    audio, sr = await preprocess_audio(audio_path)
                except Exception as e:
                    print(f"Video extraction failed, trying as audio: {str(e)}")
                    audio, sr = await preprocess_audio(file_path)
            else:
                print(f"Downloading {key} from {url}")
                file_path = await download_file(url)
                print(f"Downloaded to {file_path}")

                audio, sr = await preprocess_audio(file_path)

            print(f"Preprocessed {key}, audio length: {len(audio)}")
            audio_files[key] = (audio, sr)

        all_features = []
        for key, (audio, sr) in audio_files.items():
            print(f"Extracting features from {key}")
            features = extract_features(audio, sr)
            print(f"Extracted {len(features)} features from {key}")
            all_features.extend(features)

        print(f"Total features extracted: {len(all_features)}")
        result = predict_default(all_features)
        print(f"Prediction result: {result}")

        return result
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in predict endpoint: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Loan Default Prediction API", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
