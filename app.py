from flask import Flask, request, jsonify
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
import soundfile as sf
import io

app = Flask(__name__)

# Constants
SAMPLE_RATE = 22050  # Hz
MFCC_FEATURES = 13  # Number of MFCC features to extract
THRESHOLD = 50.0  # Distance threshold for voice matching


def extract_features_from_audio(audio_data, sample_rate=SAMPLE_RATE):
    """Extract MFCC features directly from audio data."""
    mfccs = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=MFCC_FEATURES)
    return np.mean(mfccs.T, axis=0)


def validate_audio(reference_audio, sample_audio):
    """Validate the test sample against the reference sample using MFCC features."""
    # Extract features from both reference and test samples
    reference_features = extract_features_from_audio(reference_audio)
    sample_features = extract_features_from_audio(sample_audio)

    # Calculate Euclidean distance between feature vectors
    distance = euclidean(reference_features, sample_features)

    # Determine if the voices match based on the threshold
    match = distance < THRESHOLD
    result = {
        "match": match,
        "distance": distance
    }
    return result


@app.route('/match-voice', methods=['POST'])
def match_voice():
    """API endpoint to match two voice samples."""
    try:
        # Retrieve raw audio data from the request
        reference_audio_data = request.files.get('reference').read()
        sample_audio_data = request.files.get('sample').read()

        # Convert raw audio data to numpy arrays
        reference_audio, ref_sr = sf.read(io.BytesIO(reference_audio_data))
        sample_audio, samp_sr = sf.read(io.BytesIO(sample_audio_data))

        # Resample if needed to match the expected sample rate
        if ref_sr != SAMPLE_RATE:
            reference_audio = librosa.resample(
                reference_audio, orig_sr=ref_sr, target_sr=SAMPLE_RATE)
        if samp_sr != SAMPLE_RATE:
            sample_audio = librosa.resample(
                sample_audio, orig_sr=samp_sr, target_sr=SAMPLE_RATE)

        # Validate the sample against the reference
        result = validate_audio(reference_audio, sample_audio)

        # Return the result as JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
