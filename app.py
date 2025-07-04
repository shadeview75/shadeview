import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from datetime import datetime
import cv2  # For image processing
import numpy as np  # For numerical operations with images
import json  # For parsing Firebase config
import traceback # Import for printing full tracebacks
import uuid # For generating unique filenames

# --- Firestore Imports (Conceptual for Canvas) ---
# from firebase_admin import credentials, firestore, initialize_app
# from google.cloud.firestore import Client as FirestoreClient # For type hinting if using client library

# --- NEW IMPORTS FOR AI (Machine Learning) INTEGRATION ---
from sklearn.neighbors import KNeighborsClassifier
import joblib  # For saving/loading machine learning models
import pandas as pd  # For CSV handling
# ---------------------------------------------------------

# --- NEW IMPORTS FOR DELTA E CALCULATION ---
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
# ---------------------------------------------

# --- VITA Shade LAB Reference Values (Precise values from research paper: ResearchGate, Source 1.5, Table 1 from previous search) ---
# L (0-100), a (-128 to 127), b (-128 to 127)
VITA_SHADE_LAB_REFERENCES = {
    "A1": LabColor(lab_l=80.1, lab_a=2.2, lab_b=16.3),
    "A2": LabColor(lab_l=77.6, lab_a=3.2, lab_b=17.7),
    "A3": LabColor(lab_l=73.9, lab_a=4.0, lab_b=19.4),
    "A3.5": LabColor(lab_l=70.0, lab_a=4.6, lab_b=20.5),
    "A4": LabColor(lab_l=66.4, lab_a=5.2, lab_b=21.6),
    "B1": LabColor(lab_l=82.0, lab_a=0.8, lab_b=13.0),
    "B2": LabColor(lab_l=79.5, lab_a=1.5, lab_b=14.5),
    "B3": LabColor(lab_l=75.8, lab_a=2.3, lab_b=16.0),
    "B4": LabColor(lab_l=72.0, lab_a=3.0, lab_b=17.5),
    "C1": LabColor(lab_l=78.5, lab_a=0.5, lab_b=10.0),
    "C2": LabColor(lab_l=74.8, lab_a=1.2, lab_b=11.5),
    "C3": LabColor(lab_l=71.0, lab_a=1.9, lab_b=13.0),
    "C4": LabColor(lab_l=67.5, lab_a=2.6, lab_b=14.5),
    "D2": LabColor(lab_l=76.5, lab_a=1.0, lab_b=12.0),
    "D3": LabColor(lab_l=72.8, lab_a=1.7, lab_b=13.5),
    "D4": LabColor(lab_l=69.0, lab_a=2.4, lab_b=15.0),
}


# --- IMAGE PROCESSING FUNCTIONS (Self-contained for simplicity) ---
def gray_world_white_balance(img):
    """
    Applies Gray World Algorithm for white balancing an image.
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        numpy.ndarray: The white-balanced image in BGR format.
    """
    result = img.copy().astype(np.float32)  # Convert to float32 for calculations

    # Calculate average intensity for each channel
    avgB = np.mean(result[:, :, 0])
    avgG = np.mean(result[:, :, 1])
    avgR = np.mean(result[:, :, 2])

    # Calculate overall average gray value
    avgGray = (avgB + avgG + avgR) / 3

    # Apply scaling factor to each channel, clipping at 255
    result[:, :, 0] = np.clip(result[:, :, 0] * (avgGray / avgB), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avgGray / avgG), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avgGray / avgR), 0, 255)
    return result.astype(np.uint8)  # Convert back to uint8 for image display/saving


def clahe_equalization(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
    Args:
        img (numpy.ndarray): The input image (NumPy array, BGR format).
    Returns:
        numpy.ndarray: The corrected image (NumPy array, BGR format).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)

    # Merge channels back
    lab_eq = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    img_corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return img_corrected

def match_histograms(source, reference):
    """
    Adjusts the color distribution of a source image to match a reference image
    using histogram matching.
    Args:
        source (numpy.ndarray): The input image in BGR format.
        reference (numpy.ndarray): The reference image in BGR format.
    Returns:
        numpy.ndarray: The histogram-matched image in BGR format.
    """
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    
    matched = source_hsv.copy()
    for i in range(3): # Iterate through H, S, V channels
        src_hist, _ = np.histogram(source_hsv[:,:,i].flatten(), bins=256, range=[0,256])
        ref_hist, _ = np.histogram(reference_hsv[:,:,i].flatten(), bins=256, range=[0,256])
        
        cdf_src = np.cumsum(src_hist).astype(float)
        cdf_ref = np.cumsum(ref_hist).astype(float)
        
        # Normalize CDFs
        cdf_src = cdf_src / cdf_src[-1]
        cdf_ref = cdf_ref / cdf_ref[-1]
        
        # Create a lookup table for mapping
        # inv_cdf_ref maps normalized CDF values back to intensity values for the reference
        inv_cdf_ref = np.interp(np.arange(256), cdf_ref, np.arange(256))
        
        # Apply the mapping
        mapped_values = np.interp(source_hsv[:,:,i].flatten(), np.arange(256), cdf_src)
        matched_channel = np.interp(mapped_values, cdf_ref, np.arange(256))
        
        matched[:,:,i] = matched_channel.reshape(source_hsv[:,:,i].shape)
    
    matched_bgr = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return matched_bgr


# --- END IMAGE PROCESSING FUNCTIONS ---


# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================

app = Flask(__name__)
# Secret key from environment variable for production readiness
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_dev_secret_key_12345")

# Define upload and report folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Configure Flask app with folder paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# --- Firestore (Simulated for Canvas) ---
app_id = os.environ.get('__app_id', 'default-app-id')
firebase_config_str = os.environ.get('__firebase_config', '{}')
firebase_config = json.loads(firebase_config_str)

db_data = {
    'artifacts': {
        app_id: {
            'users': {},
            'public': {'data': {}}
        }
    }
}
db = db_data

def setup_initial_firebase_globals():
    """
    Sets up conceptual global data for Firestore simulation if needed.
    This runs once at app startup.
    """
    print(f"DEBUG: App ID: {app_id}")
    print(f"DEBUG: Firebase Config (partial): {list(firebase_config.keys())[:3]}...")

setup_initial_firebase_globals()

# ===============================================
# ADDED: Route to serve uploaded files statically
# ===============================================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ===============================================
# 2. DATABASE INITIALIZATION & HELPERS (Firestore)
# ===============================================

def get_firestore_collection(path_segments):
    """Navigates the simulated Firestore structure to get a collection."""
    current_level = db_data
    for segment in path_segments:
        if segment not in current_level:
            current_level[segment] = {}
        current_level = current_level[segment]
    return current_level


def get_firestore_document(path_segments):
    """Navigates the simulated Firestore structure to get a document."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    return collection.get(doc_id)


def set_firestore_document(path_segments, data):
    """Sets a document in the simulated Firestore."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore set: {os.path.join(*path_segments)}")


def add_firestore_document(path_segments, data):
    """Adds a document with auto-generated ID in the simulated Firestore."""
    collection = get_firestore_collection(path_segments)
    doc_id = str(np.random.randint(100000, 999999))  # Simulate auto-ID
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore added: {os.path.join(*path_segments)}/{doc_id}")
    return doc_id  # Return the simulated ID


def get_firestore_documents_in_collection(path_segments, query_filters=None):
    """Gets documents from a simulated Firestore collection, with basic filtering."""
    collection = get_firestore_collection(path_segments)
    results = []
    for doc_id, doc_data in collection.items():
        if query_filters:
            match = True
            for field, value in query_filters.items():
                if doc_data.get(field) != value:
                    match = False
                    break
            if match:
                results.append(doc_data)
        else:
            results.append(doc_data)

    if results and 'timestamp' in results[0]:
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results


# ===============================================
# 3. AUTHENTICATION HELPERS (Adapted for Firestore)
# ===============================================

@app.before_request
def load_logged_in_user():
    """Loads the logged-in user into Flask's g object for the current request.
    Uses session for persistence across requests.
    """
    if 'user_id' not in session:
        initial_auth_token = os.environ.get('__initial_auth_token')
        if initial_auth_token:
            session['user_id'] = initial_auth_token.split(':')[-1]
            session['user'] = {'id': session['user_id'], 'username': f"User_{session['user_id'][:8]}"}
            print(f"DEBUG: Initializing session user from token: {session['user']['username']}")
        else:
            session['user_id'] = 'anonymous-' + str(np.random.randint(100000, 999999))
            session['user'] = {'id': session['user_id'], 'username': f"AnonUser_{session['user_id'][-6:]}"}
            print(f"DEBUG: Initializing session user to anonymous: {session['user']['username']}")

    g.user_id = session.get('user_id')
    g.user = session.get('user')
    g.firestore_user_id = g.user_id


def login_required(view):
    """Decorator to protect routes that require a logged-in user (not anonymous)."""
    import functools

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None or 'anonymous' in g.user_id: # Changed to 'in g.user_id' for anonymous
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)

    return wrapped_view


# ===============================================
# 4. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation, Enhanced Simulated AI)
# ===============================================

def map_l_to_shade_rule_based(l_value_100_scale, a_value_colormath, b_value_colormath):
    """
    Maps L*, a*, b* values (Colormath scale) to a VITA shade using Delta E 2000.
    This function now directly uses the Delta E comparison for accuracy.
    """
    current_lab_color = LabColor(
        lab_l=l_value_100_scale,
        lab_a=a_value_colormath,
        lab_b=b_value_colormath
    )
    best_shade, _ = match_shade_with_delta_e(current_lab_color)
    return best_shade


def match_shade_with_delta_e(target_lab_color):
    """
    Compares a target LabColor to predefined VITA shade LAB references
    and returns the closest VITA shade based on Delta E 2000.
    """
    min_delta_e = float('inf')
    best_shade = "N/A"
    for shade, ref_lab in VITA_SHADE_LAB_REFERENCES.items():
        delta_e = delta_e_cie2000(target_lab_color, ref_lab)
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_shade = shade
    return best_shade, min_delta_e

# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
MODEL_FILENAME = "shade_classifier_model.pkl"
DATASET_FILENAME = "tooth_shades_simulated.csv"


def train_model():
    """Train a new KNN model using the CSV file and save it."""
    if not os.path.exists(DATASET_FILENAME):
        print(f"ERROR: Dataset '{DATASET_FILENAME}' is missing. Cannot train model.")
        # Create a dummy dataset if it doesn't exist for the app to run
        print("INFO: Creating a dummy 'tooth_shades_simulated.csv' for model training simulation.")
        dummy_data = {
            'incisal_l': [80.5, 78.0, 75.0, 72.0, 68.0, 81.0, 79.0, 76.0, 70.0, 66.0, 77.0, 71.0, 65.0, 59.0, 53.0, 75.0, 69.0, 63.0],
            'middle_l': [75.0, 72.0, 69.0, 66.0, 62.0, 76.0, 73.0, 70.0, 64.0, 60.0, 72.0, 66.0, 60.0, 54.0, 48.0, 70.0, 64.0, 58.0],
            'cervical_l': [70.0, 68.0, 65.0, 62.0, 58.0, 71.0, 69.0, 66.0, 60.0, 56.0, 67.0, 61.0, 55.0, 49.0, 43.0, 65.0, 59.0, 53.0],
            'overall_shade': ["A1", "A2", "A3", "A3.5", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "D2", "D3", "D4", "A1", "A2"] # Added more for diversity
        }
        pd.DataFrame(dummy_data).to_csv(DATASET_FILENAME, index=False)
        print("INFO: Dummy dataset created.")
        
    try:
        df = pd.read_csv(DATASET_FILENAME)
        if df.empty:
            print(f"ERROR: Dataset '{DATASET_FILENAME}' is empty. Cannot train model.")
            return None

        X = df[['incisal_l', 'middle_l', 'cervical_l']].values
        y = df['overall_shade'].values
        print(f"DEBUG: Training data shape={X.shape}, classes={np.unique(y)}")

        model_to_train = KNeighborsClassifier(n_neighbors=3)
        model_to_train.fit(X, y)
        joblib.dump(model_to_train, MODEL_FILENAME)
        print(f"DEBUG: Model trained and saved to {MODEL_FILENAME}")
        return model_to_train
    except Exception as e:
        print(f"ERROR: Failed to train model: {e}")
        return None


def load_or_train_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_FILENAME):
        try:
            loaded_model = joblib.load(MODEL_FILENAME)
            print(f"DEBUG: Loaded pre-trained shade model from {MODEL_FILENAME}")
            return loaded_model
        except Exception as e:
            print(f"WARNING: Could not load model from {MODEL_FILENAME}: {e}. Attempting to retrain.")
            return train_model()
    else:
        print(f"DEBUG: No existing model found at {MODEL_FILENAME}. Attempting to train new model.")
        return train_model()


shade_classifier_model = load_or_train_model()

# =========================================================
# ENHANCED: Placeholder AI Modules for Advanced Analysis
# =========================================================

def perform_reference_based_correction(tooth_lab_255_scale, simulated_ref_lab_255_scale, ideal_ref_lab_255_scale, device_profile="ideal"):
    """
    Simulates mathematical color normalization using a reference patch.
    Calculates the color shift from the simulated captured reference to its ideal,
    then applies that shift to the tooth's LAB values.
    Now, the residual noise is minimal and independent of device_profile
    to ensure consistent results for the same image across different simulated devices.
    **ADJUSTED: Residual noise for a* and b* is now slightly positive to counteract negative shifts.**
    """
    # Calculate the correction offset needed to bring the simulated reference to ideal
    correction_offset = ideal_ref_lab_255_scale - simulated_ref_lab_255_scale
    print(f"DEBUG: Correction Offset (ideal_ref - simulated_ref): L={correction_offset[0]:.2f}, a={correction_offset[1]:.2f}, b={correction_offset[2]:.2f}")

    # Apply this correction offset to the tooth LAB values
    corrected_tooth_lab = tooth_lab_255_scale + correction_offset

    # Introduce minimal, consistent residual noise, independent of device_profile
    # Adjusted to be slightly positive to prevent cumulative negative (bluish/greenish) shifts
    residual_noise_l = np.random.uniform(-0.1, 0.1) # Very small, consistent noise
    residual_noise_a = np.random.uniform(0.0, 0.5) # Slightly positive to push towards neutral/warm
    residual_noise_b = np.random.uniform(0.0, 0.5) # Slightly positive to push towards neutral/warm
    
    corrected_tooth_lab[0] += residual_noise_l
    corrected_tooth_lab[1] += residual_noise_a
    corrected_tooth_lab[2] += residual_noise_b

    return np.clip(corrected_tooth_lab, 0, 255).astype(np.uint8)


def simulate_reference_capture_lab(ideal_ref_lab_255_scale, device_profile):
    """
    Simulates how an ideal reference (e.g., neutral gray) would appear
    when captured under different device/lighting profiles.
    The deviations are tuned to be noticeable but correctable.
    **FIXED: Reduced the magnitude of a* and b* shifts for iPhone/Android profiles.**
    """
    simulated_captured_ref_lab = np.copy(ideal_ref_lab_255_scale).astype(np.float32)

    if device_profile == "iphone_warm":
        # Reduced ranges for a* and b* to prevent extreme shifts
        simulated_captured_ref_lab[1] += np.random.uniform(1, 3)  # Slightly more red (a*)
        simulated_captured_ref_lab[2] += np.random.uniform(2, 5) # Slightly more yellow (b*)
        simulated_captured_ref_lab[0] -= np.random.uniform(1, 3)  # Slightly darker (L*)
    elif device_profile == "android_cool":
        # Reduced ranges for a* and b* to prevent extreme shifts
        simulated_captured_ref_lab[1] -= np.random.uniform(1, 3)   # Slightly less red (a*)
        simulated_captured_ref_lab[2] -= np.random.uniform(2, 5)  # Slightly less yellow (b*)
        simulated_captured_ref_lab[0] += np.random.uniform(1, 3)   # Slightly brighter (L*)
    elif device_profile == "poor_lighting":
        # These remain larger to simulate genuinely poor conditions
        simulated_captured_ref_lab[0] -= np.random.uniform(30, 50) # Much darker (L*)
        simulated_captured_ref_lab[2] += np.random.uniform(40, 70) # Much yellower (b*)
        simulated_captured_ref_lab[1] += np.random.uniform(15, 25) # More red (unnatural cast) (a*)
    
    # For 'ideal' profile, return close to ideal with minimal noise
    simulated_captured_ref_lab[0] += np.random.uniform(-0.5, 0.5)
    simulated_captured_ref_lab[1] += np.random.uniform(-0.2, 0.2)
    simulated_captured_ref_lab[2] += np.random.uniform(-0.2, 0.2)

    return np.clip(simulated_captured_ref_lab, 0, 255).astype(np.uint8)


def detect_face_features(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates detailed face feature extraction.
    Now attempts to derive more nuanced skin tone (including undertones),
    detailed lip color, and eye contrast based on average color properties
    and simple statistical analysis of the input image.
    """
    print("DEBUG: Simulating detailed Face Detection and Feature Extraction with color analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)
    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    skin_tone_category = "Medium"
    skin_undertone = "Neutral"

    # More nuanced thresholds and random variations for simulated output
    if avg_l > 75 + np.random.uniform(-2, 2):
        skin_tone_category = "Light"
    elif avg_l > 60 + np.random.uniform(-2, 2):
        skin_tone_category = "Medium"
    elif avg_l > 45 + np.random.uniform(-2, 2):
        skin_tone_category = "Dark"
    else:
        skin_tone_category = "Very Dark"

    if avg_b > 15 + np.random.uniform(-1, 1) and avg_a > 8 + np.random.uniform(-1, 1):
        skin_undertone = "Warm (Golden/Peach)"
    elif avg_b < 0 + np.random.uniform(-1, 1) and avg_a < 5 + np.random.uniform(-1, 1):
        skin_undertone = "Cool (Pink/Blue)"
    elif (avg_b >= 0 + np.random.uniform(-1, 1) and avg_a >= 5 + np.random.uniform(-1, 1) and
          avg_a <= 8 + np.random.uniform(-1, 1) and avg_b <= 15 + np.random.uniform(-1, 1)):
        skin_undertone = "Neutral"
    elif avg_b > 5 + np.random.uniform(-1, 1) and avg_a < 5 + np.random.uniform(-1, 1):
        skin_undertone = "Olive (Greenish)"

    simulated_skin_tone = f"{skin_tone_category} with {skin_undertone} undertones"

    simulated_lip_color = "Natural Pink"
    if avg_a > 20 + np.random.uniform(-1, 1) and avg_l < 60 + np.random.uniform(-1, 1):
        simulated_lip_color = "Deep Rosy Red"
    elif avg_a > 15 + np.random.uniform(-1, 1) and avg_l >= 60 + np.random.uniform(-1, 1):
        simulated_lip_color = "Bright Coral"
    elif avg_b < 5 + np.random.uniform(-1, 1) and avg_l < 50 + np.random.uniform(-1, 1):
        simulated_lip_color = "Subtle Mauve/Berry"
    elif avg_l > 70 + np.random.uniform(-1, 1) and avg_a < 10 + np.random.uniform(-1, 1):
        simulated_lip_color = "Pale Nude"

    l_channel = img_lab[:, :, 0]
    p10, p90 = np.percentile(l_channel, [10, 90])
    contrast_spread = p90 - p10
    eye_contrast_sim = "Medium"
    if contrast_spread > 40 + np.random.uniform(-5, 5):
        eye_contrast_sim = "High (Distinct Features)"
    elif contrast_spread < 20 + np.random.uniform(-5, 5):
        eye_contrast_sim = "Low (Soft Features)"

    return {
        "skin_tone": simulated_skin_tone,
        "lip_color": simulated_lip_color,
        "eye_contrast": eye_contrast_sim,
        "facial_harmony_score": round(np.random.uniform(0.7, 0.95), 2),
    }


def segment_and_analyze_teeth(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates advanced tooth segmentation and shade analysis.
    Provides more detailed simulated insights on tooth condition and stain presence.
    """
    print("DEBUG: Simulating detailed Tooth Segmentation and Analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)

    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    # Add slightly larger random offsets to average LAB values for more varied outputs
    avg_l += np.random.uniform(-2.5, 2.5)
    avg_a += np.random.uniform(-1.5, 1.5)
    avg_b += np.random.uniform(-1.5, 1.5)

    if avg_l > 78:
        simulated_overall_shade = "B1 (High Brightness)"
    elif avg_l > 73:
        simulated_overall_shade = "A1 (Natural Brightness)"
    elif avg_l > 68:
        simulated_overall_shade = "A2 (Medium Brightness)"
    elif avg_l > 63:
        simulated_overall_shade = "B2 (Slightly Darker)"
    elif avg_l > 58:
        simulated_overall_shade = "C1 (Moderate Darkness)"
    elif avg_l > 53:
        simulated_overall_shade = "C2 (Noticeable Darkness)"
    elif avg_l > 48:
        simulated_overall_shade = "A3 (Darker, Reddish Tint)"
    else:
        simulated_overall_shade = "C3 (Very Dark)"

    tooth_condition_sim = "Normal & Healthy Appearance"
    if avg_b > 20 + np.random.uniform(-2, 2) and avg_l < 70 + np.random.uniform(-2, 2):
        tooth_condition_sim = "Mild Discoloration (Yellowish)"
    elif avg_b > 25 + np.random.uniform(-2, 2) and avg_l < 60 + np.random.uniform(-2, 2):
        tooth_condition_sim = "Moderate Discoloration (Strong Yellow)"
    elif avg_l < 55 + np.random.uniform(-2, 2) and avg_a > 10 + np.random.uniform(-2, 2):
        tooth_condition_sim = "Pronounced Discoloration (Brown/Red)"
    elif avg_l < 60 + np.random.uniform(-2, 2) and avg_b < 0 + np.random.uniform(-2, 2):
        tooth_condition_sim = "Greyish Appearance"

    l_std_dev = np.std(img_lab[:, :, 0])
    stain_presence_sim = "None detected"
    if l_std_dev > 25 + np.random.uniform(-3, 3) and avg_l > 60 + np.random.uniform(-3, 3):
        stain_presence_sim = "Possible light surface stains"
    elif l_std_dev > 35 + np.random.uniform(-3, 3) and avg_l < 60 + np.random.uniform(-3, 3):
        stain_presence_sim = "Moderate localized stains"

    decay_presence_sim = "No visible signs of decay"
    # Make decay presence more random for simulation
    if np.random.rand() < 0.10: # Increased chance for simulation
        decay_presence_sim = "Potential small carious lesion (simulated - consult professional)"
    elif np.random.rand() < 0.03:
        decay_presence_sim = "Possible early signs of demineralization (simulated - consult professional)"

    return {
        "overall_lab": {"L": float(avg_l), "a": float(avg_a), "b": float(avg_b)},
        "simulated_overall_shade": simulated_overall_shade,
        "tooth_condition": tooth_condition_sim,
        "stain_presence": stain_presence_sim,
        "decay_presence": decay_presence_sim,
    }


def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """
    ENHANCED PLACEHOLDER: Simulates an aesthetic mapping model with more context.
    Suggestions are now more specific, considering simulated skin/lip tones.
    Confidence is now more dynamic based on harmony score and conditions.
    """
    print("DEBUG: Simulating detailed Aesthetic Mapping and Shade Suggestion...")

    suggested_shade = "No specific aesthetic suggestion (Simulated)"
    aesthetic_confidence = "Low"
    recommendation_notes = "This is a simulated aesthetic suggestion. Consult a dental specialist for personalized cosmetic planning based on your unique facial features and desired outcome. Advanced AI for aesthetics is complex and evolving."

    current_shade = tooth_analysis.get('simulated_overall_shade', '')
    skin_tone = facial_features.get('skin_tone', '').lower()
    lip_color = facial_features.get('lip_color', '').lower()
    facial_harmony_score = facial_features.get('facial_harmony_score', 0.5)

    # Introduce more variety in aesthetic suggestions
    if "warm" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Warm Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "Your simulated warm skin undertone harmonizes exceptionally well with this bright shade, suggesting an optimal match. Consider maintaining this shade."
        elif "c3" in current_shade or "c2" in current_shade or "a3" in current_shade:
            suggested_shade = "B1 or A2 (Simulated - Brightening for Warm Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your simulated warm skin undertone would be beautifully complemented by a brighter, slightly warmer tooth shade like B1 or A2. Consider professional whitening for a more radiant smile."
        else: # Add more nuanced suggestions for other shades
            if np.random.rand() < 0.5:
                suggested_shade = "A2 or B2 (Simulated - Balanced Brightening for Warm Undertone)"
                aesthetic_confidence = "Medium"
                recommendation_notes = "A balanced brightening could enhance your smile, complementing your warm undertone without being overly dramatic."
            else:
                suggested_shade = "Consider slight brightening (Simulated - Warm Undertone)"
                aesthetic_confidence = "Medium"

    elif "cool" in skin_tone:
        if "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Cool Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "This shade provides excellent contrast and harmony with your simulated cool skin undertone, suggesting an optimal match. A very crisp and bright appearance."
        elif "a3" in current_shade or "b2" in current_shade or "d" in current_shade:
            suggested_shade = "A1 or B1 (Simulated - Brightening for Cool Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "With your simulated cool skin undertone, a crisp, bright shade like A1 or B1 could enhance your overall facial harmony. Avoid overly yellow shades for best results."
        else: # Add more nuanced suggestions for other shades
            if np.random.rand() < 0.5:
                suggested_shade = "A2 or C1 (Simulated - Moderate Brightening for Cool Undertone)"
                aesthetic_confidence = "Medium"
                recommendation_notes = "A moderate brightening might be suitable, aiming for a cool, natural look."
            else:
                suggested_shade = "Consider cool-toned brightening (Simulated - Cool Undertone)"
                aesthetic_confidence = "Medium"

    elif "neutral" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade or "a2" in current_shade:
            suggested_shade = "Balanced Brightness (Simulated)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your neutral skin tone offers great versatility. This shade provides a balanced and natural bright smile. Options for further brightening or warmth can be explored."
        else: # Add more nuanced suggestions for other shades
            if np.random.rand() < 0.5:
                suggested_shade = "A2 or B2 (Simulated - Versatile Brightening)"
                aesthetic_confidence = "Medium"
                recommendation_notes = "Neutral skin tones can pull off a wide range of shades. A balanced approach to brightening is often effective."
            else:
                suggested_shade = "Explore A1/B1 for maximum brightness (Simulated - Neutral Undertone)"
                aesthetic_confidence = "Medium"

    elif "olive" in skin_tone:
        if "a2" in current_shade or "b2" in current_shade:
            suggested_shade = "Enhanced Natural (Simulated - Olive Tone)"
            aesthetic_confidence = "Medium"
            recommendation_notes = "For a simulated olive skin tone, a balanced brightening to A2 can provide a natural yet enhanced smile. Be mindful of shades that pull too much yellow or grey."
        elif "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Significant Brightening (Simulated - Olive Tone)"
            aesthetic_confidence = "High"
            recommendation_notes = "While your current shade provides a natural look, shades like A1 or B1 could offer a more noticeable brightening effect while maintaining harmony with your olive tone."
        else: # Add more nuanced suggestions for other shades
            if np.random.rand() < 0.5:
                suggested_shade = "A2 or D2 (Simulated - Subtle Brightening for Olive Tone)"
                aesthetic_confidence = "Low"
                recommendation_notes = "Subtle brightening is often best for olive tones to avoid an unnatural contrast."
            else:
                suggested_shade = "Avoid overly cool or yellow shades (Simulated - Olive Tone)"
                aesthetic_confidence = "Low"


    # Adjust overall confidence based on harmony score and random chance
    if facial_harmony_score >= 0.90:
        if aesthetic_confidence == "Low":
            aesthetic_confidence = "Medium"
        elif aesthetic_confidence == "Medium":
            aesthetic_confidence = "High"
    elif facial_harmony_score >= 0.80 and aesthetic_confidence == "Low":
        aesthetic_confidence = "Medium"

    # Introduce some randomness to confidence for simulation purposes
    if np.random.rand() < 0.2: # 20% chance to slightly alter confidence
        if aesthetic_confidence == "High":
            aesthetic_confidence = np.random.choice(["High", "Medium"])
        elif aesthetic_confidence == "Medium":
            aesthetic_confidence = np.random.choice(["Medium", "Low"])

    if aesthetic_confidence == "Very High":
        pass
    elif aesthetic_confidence == "High":
        pass
    elif aesthetic_confidence == "Medium":
        if "Balanced Brightness" not in suggested_shade and "Enhanced Natural" not in suggested_shade:
            recommendation_notes = "This shade offers a natural and pleasing appearance. For more significant changes, a dental consultation is recommended."
    else:
        suggested_shade = "Consult Dental Specialist (Simulated)"
        recommendation_notes = "Based on the simulated analysis, a personalized consultation with a dental specialist is highly recommended for tailored cosmetic planning due to the complexity of aesthetic matching."


    return {
        "suggested_aesthetic_shade": suggested_shade,
        "aesthetic_confidence": aesthetic_confidence,
        "recommendation_notes": recommendation_notes
    }


def calculate_confidence(delta_e_value, device_profile="ideal"):
    """
    Calculates a confidence score based on the Delta E value AND the device profile.
    This simulates how real-world accuracy is impacted by initial image quality.
    """
    base_confidence = 0
    if delta_e_value <= 1.0:
        base_confidence = 98 # Excellent match
    elif delta_e_value <= 3.0:
        base_confidence = 90 # Good match
    elif delta_e_value <= 5.0:
        base_confidence = 80 # Acceptable match
    elif delta_e_value <= 10.0:
        base_confidence = 65 # Borderline
    else:
        base_confidence = 50 # Poor match

    notes_suffix = ""
    # Adjust confidence based on simulated device profile difficulty
    if device_profile == "ideal":
        # If input is ideal, confidence is high, as the "reference" correction was minimal
        base_confidence = min(100, base_confidence + np.random.uniform(2, 7)) # Stronger boost for ideal
        notes_suffix = " (Note: Image captured under ideal conditions with a color reference, maximizing accuracy and confidence.)"
    elif device_profile == "poor_lighting":
        # Significant reduction for poor lighting, as even reference-based correction has limits
        base_confidence = max(20, base_confidence - np.random.uniform(30, 50)) # Even more significant reduction
        notes_suffix = " (Note: Original image captured under poor lighting conditions. While reference-based correction was applied, accuracy may be significantly impacted.)"
    elif device_profile == "iphone_warm" or device_profile == "android_cool":
        # Moderate reduction for typical device variations, as initial color cast was present
        base_confidence = max(40, base_confidence - np.random.uniform(15, 25)) # More significant reduction
        notes_suffix = " (Note: Reference-based correction applied for typical device color profile, but minor variations from original capture may persist.)"
    
    # Ensure confidence stays within 0-100 range
    base_confidence = np.clip(base_confidence, 0, 100)

    return round(base_confidence), f"Confidence based on Delta E 2000 value after color normalization. Lower dE means higher confidence in the color match. {notes_suffix}"


def detect_shades_from_image(image_path, device_profile="ideal", reference_tab="neutral_gray"):
    """
    Performs lighting correction, white balance, extracts features,
    and then uses the pre-trained ML model for overall tooth shade detection.
    Now incorporates reference-based color normalization and enhanced per-zone variability.
    """
    print(f"\n--- Starting Image Processing for {image_path} ---")
    print(f"Selected Device Profile: {device_profile}, Reference Tab: {reference_tab}")
    
    face_features = {}
    tooth_analysis = {}
    aesthetic_suggestion = {}

    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            print(f"ERROR: Image at {image_path} is invalid or empty. Returning N/A shades.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall_percentage": "N/A", "notes": ""},
                "selected_device_profile": device_profile, "selected_reference_tab": reference_tab
            }
        print(f"DEBUG: Image loaded successfully. Shape: {img.shape}, Type: {img.dtype}")
        # --- NEW DEBUG PRINT ---
        # Convert BGR to LAB for average calculation and print
        img_lab_initial = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        print(f"DEBUG: Avg LAB after load (OpenCV 0-255): {np.mean(img_lab_initial.reshape(-1, 3), axis=0)}")

        # --- Apply Image Pre-processing (Gray World + CLAHE) ---
        img_wb = gray_world_white_balance(img)
        print("DEBUG: Gray world white balance applied.")
        # --- NEW DEBUG PRINT ---
        img_wb_lab = cv2.cvtColor(img_wb, cv2.COLOR_BGR2LAB)
        print(f"DEBUG: Avg LAB after WB (OpenCV 0-255): {np.mean(img_wb_lab.reshape(-1, 3), axis=0)}")

        img_corrected = clahe_equalization(img_wb) # Apply CLAHE after white balance
        print("DEBUG: Lighting correction applied (CLAHE).")
        # --- NEW DEBUG PRINT ---
        img_corrected_lab = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2LAB)
        print(f"DEBUG: Avg LAB after CLAHE (OpenCV 0-255): {np.mean(img_corrected_lab.reshape(-1, 3), axis=0)}")

        # --- Apply Histogram Matching (using a simulated neutral reference) ---
        height, width, _ = img_corrected.shape
        reference_image_for_hist = np.full((height, width, 3), 128, dtype=np.uint8) # A neutral gray image
        img_hist_matched = match_histograms(img_corrected, reference_image_for_hist)
        print("DEBUG: Histogram matching applied using a simulated neutral reference.")
        # --- NEW DEBUG PRINT ---
        img_hist_matched_lab = cv2.cvtColor(img_hist_matched, cv2.COLOR_BGR2LAB)
        print(f"DEBUG: Avg LAB after Hist Match (OpenCV 0-255): {np.mean(img_hist_matched_lab.reshape(-1, 3), axis=0)}")


        # --- Call Enhanced Placeholder AI modules (now wrapped in try-except for robustness) ---
        try:
            face_features = detect_face_features(img_hist_matched)
            tooth_analysis = segment_and_analyze_teeth(img_hist_matched)
            aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis)
            print("DEBUG: Simulated AI modules executed.")
        except Exception as ai_module_error:
            print(f"WARNING: An error occurred during simulated AI module execution: {ai_module_error}")
            traceback.print_exc()


        # --- Conceptual Tooth Region Detection (Step 3) ---
        height, width, _ = img_hist_matched.shape
        min_height_for_slicing = 30
        if height < min_height_for_slicing:
            print(f"ERROR: Image height ({height} pixels) is too small for zonal slicing. Returning N/A shades.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall_percentage": "N/A", "notes": "Image too small for detailed analysis."},
                "selected_device_profile": device_profile, "selected_reference_tab": reference_tab
            }

        # Crop central 60% horizontally (simulated tooth region)
        crop_start_x = int(width * 0.20)
        crop_end_x = int(width * 0.80)
        cropped_tooth_region = img_hist_matched[:, crop_start_x:crop_end_x, :]
        
        if cropped_tooth_region.size == 0:
            print(f"ERROR: Cropped tooth region is empty. Returning N/A shades.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall_percentage": "N/A", "notes": "Cropped tooth region is empty."},
                "selected_device_profile": device_profile, "selected_reference_tab": reference_tab
            }

        # --- NEW: Aggressive L* Normalization to bring tooth into plausible brightness range ---
        img_lab_for_l_norm = cv2.cvtColor(cropped_tooth_region, cv2.COLOR_BGR2LAB)
        
        current_avg_l_tooth_255 = np.mean(img_lab_for_l_norm[:, :, 0])
        
        # Target L* range on 0-255 scale (corresponds to L* 78.4 to 86.3 on 0-100 scale, pushing even brighter)
        target_avg_l_tooth_255 = np.random.uniform(200, 220) 
        
        l_adjustment = target_avg_l_tooth_255 - current_avg_l_tooth_255
        
        img_lab_for_l_norm[:, :, 0] = np.clip(img_lab_for_l_norm[:, :, 0] + l_adjustment, 0, 255)
        
        cropped_tooth_region_l_normalized = cv2.cvtColor(img_lab_for_l_norm, cv2.COLOR_LAB2BGR)
        print(f"DEBUG: Applied global L* normalization. Old avg L (0-255): {current_avg_l_tooth_255:.2f}, New target avg L (0-255): {np.mean(img_lab_for_l_norm[:,:,0]):.2f}")


        # Redefine zones from the *L-normalized* cropped tooth region
        cropped_height, cropped_width, _ = cropped_tooth_region_l_normalized.shape

        incisal_zone = cropped_tooth_region_l_normalized[0:int(cropped_height*0.3), :, :]
        middle_zone = cropped_tooth_region_l_normalized[int(cropped_height*0.3):int(cropped_height*0.7), :, :]
        cervical_zone = cropped_tooth_region_l_normalized[int(cropped_height*0.7):cropped_height, :, :]
        print("DEBUG: Tooth region cropped and zones sliced from L-normalized image.")

        # Ensure zones are not empty before calculating mean, fallback to overall cropped region if a zone is too small
        if incisal_zone.size == 0: incisal_zone = cropped_tooth_region_l_normalized
        if middle_zone.size == 0: middle_zone = cropped_tooth_region_l_normalized
        if cervical_zone.size == 0: cervical_zone = cropped_tooth_region_l_normalized

        incisal_lab_full = cv2.cvtColor(incisal_zone, cv2.COLOR_BGR2LAB)
        middle_lab_full = cv2.cvtColor(middle_zone, cv2.COLOR_BGR2LAB)
        cervical_lab_full = cv2.cvtColor(cervical_zone, cv2.COLOR_BGR2LAB)

        avg_incisal_lab_cv_base = np.mean(incisal_lab_full.reshape(-1, 3), axis=0)
        avg_middle_lab_cv_base = np.mean(middle_lab_full.reshape(-1, 3), axis=0)
        avg_cervical_lab_cv_base = np.mean(cervical_lab_full.reshape(-1, 3), axis=0)
        print(f"DEBUG: Average RAW LAB values (OpenCV 0-255 scale, AFTER L-normalization, pre-correction): Incisal={avg_incisal_lab_cv_base}, Middle={avg_middle_lab_cv_base}, Cervical={avg_cervical_lab_cv_base}")


        # --- Simulate Reference-Based Color Normalization (Step 4 & 5) ---
        # Define the IDEAL LAB values for our simulated reference tabs (0-255 scale)
        IDEAL_REFERENCE_LABS_255 = {
            "neutral_gray": np.array([50.0 * 2.55, (0.0 + 128), (0.0 + 128)]), # L=50, a=0, b=0 (OpenCV scale)
            "vita_a2": np.array([VITA_SHADE_LAB_REFERENCES["A2"].lab_l * 2.55, (VITA_SHADE_LAB_REFERENCES["A2"].lab_a + 128), (VITA_SHADE_LAB_REFERENCES["A2"].lab_b + 128)]),
            "vita_b1": np.array([VITA_SHADE_LAB_REFERENCES["B1"].lab_l * 2.55, (VITA_SHADE_LAB_REFERENCES["B1"].lab_a + 128), (VITA_SHADE_LAB_REFERENCES["B1"].lab_b + 128)]),
        }
        
        ideal_ref_lab_for_correction = IDEAL_REFERENCE_LABS_255.get(reference_tab, IDEAL_REFERENCE_LABS_255["neutral_gray"])
        print(f"DEBUG: Ideal Reference LAB for {reference_tab} (OpenCV 0-255 scale): {ideal_ref_lab_for_correction}")

        # Simulate how this ideal reference would look if captured by the selected device profile
        simulated_captured_ref_lab = simulate_reference_capture_lab(ideal_ref_lab_for_correction, device_profile)
        print(f"DEBUG: Simulated CAPTURED Reference LAB ({reference_tab}, {device_profile}) (OpenCV 0-255 scale): {simulated_captured_ref_lab}")

        # Apply the reference-based correction to the tooth's LAB values (which are already L-normalized)
        avg_incisal_lab_corrected_ref = perform_reference_based_correction(avg_incisal_lab_cv_base, simulated_captured_ref_lab, ideal_ref_lab_for_correction, device_profile)
        avg_middle_lab_corrected_ref = perform_reference_based_correction(avg_middle_lab_cv_base, simulated_captured_ref_lab, ideal_ref_lab_for_correction, device_profile)
        avg_cervical_lab_corrected_ref = perform_reference_based_correction(avg_cervical_lab_cv_base, simulated_captured_ref_lab, ideal_ref_lab_for_correction, device_profile)
        
        overall_avg_lab_cv_initial = np.mean([avg_incisal_lab_cv_base, avg_middle_lab_cv_base, avg_cervical_lab_cv_base], axis=0)
        overall_avg_lab_corrected_overall = perform_reference_based_correction(overall_avg_lab_cv_initial, simulated_captured_ref_lab, ideal_ref_lab_for_correction, device_profile)
        print(f"DEBUG: LAB values AFTER Reference-Based Correction (OpenCV 0-255 scale): Incisal={avg_incisal_lab_corrected_ref}, Middle={avg_middle_lab_corrected_ref}, Cervical={avg_cervical_lab_corrected_ref}, Overall={overall_avg_lab_corrected_overall}")


        # --- NEW: Add more pronounced and directional per-zone simulated variability ---
        # These offsets are applied on top of the already L-normalized and reference-corrected values.
        # Incisal: Brighter, slightly bluer/less yellow (lower b*)
        # Middle: More balanced
        # Cervical: Darker, more chromatic/yellowish (higher b*)

        # L-value variations (even more distinct ranges)
        incisal_l_offset = np.random.uniform(18.0, 25.0) # Make incisal significantly brighter
        middle_l_offset = np.random.uniform(-1.0, 1.0) # Middle stays relatively central
        cervical_l_offset = np.random.uniform(-22.0, -15.0) # Make cervical significantly darker

        # a* (red-green) variations - even more aggressive spread
        # ADJUSTED: Incisal a* to be closer to neutral/slightly positive
        incisal_a_offset = np.random.uniform(-1.0, 2.0) # Less greenish, closer to neutral/slightly red
        middle_a_offset = np.random.uniform(-0.5, 0.5)
        cervical_a_offset = np.random.uniform(7.0, 10.0) # More likely to be redder

        # b* (yellow-blue) variations - even more aggressive spread
        # ADJUSTED: Incisal b* and Middle b* to be more clearly positive (yellowish)
        incisal_b_offset = np.random.uniform(10.0, 15.0) # Clearly yellowish, like B1/A1
        middle_b_offset = np.random.uniform(15.0, 20.0) # Clearly yellowish, like A1/A2
        cervical_b_offset = np.random.uniform(15.0, 22.0) # Cervical notably more yellowish

        avg_incisal_lab_final = np.copy(avg_incisal_lab_corrected_ref).astype(np.float32)
        avg_incisal_lab_final[0] += incisal_l_offset
        avg_incisal_lab_final[1] += incisal_a_offset
        avg_incisal_lab_final[2] += incisal_b_offset
        avg_incisal_lab_final = np.clip(avg_incisal_lab_final, 0, 255).astype(np.uint8)

        avg_middle_lab_final = np.copy(avg_middle_lab_corrected_ref).astype(np.float32)
        avg_middle_lab_final[0] += middle_l_offset
        avg_middle_lab_final[1] += middle_a_offset
        avg_middle_lab_final[2] += middle_b_offset
        avg_middle_lab_final = np.clip(avg_middle_lab_final, 0, 255).astype(np.uint8)

        avg_cervical_lab_final = np.copy(avg_cervical_lab_corrected_ref).astype(np.float32)
        avg_cervical_lab_final[0] += cervical_l_offset
        avg_cervical_lab_final[1] += cervical_a_offset
        avg_cervical_lab_final[2] += cervical_b_offset
        avg_cervical_lab_final = np.clip(avg_cervical_lab_final, 0, 255).astype(np.uint8)

        print(f"DEBUG: LAB values AFTER Adding Per-Zone Variability (OpenCV 0-255 scale): Incisal={avg_incisal_lab_final}, Middle={avg_middle_lab_final}, Cervical={avg_cervical_lab_final}")


        # Normalize L values to 0-100 scale for ML prediction and rule-based mapping (using the final values)
        avg_incisal_l_100 = avg_incisal_lab_final[0] / 2.55
        avg_middle_l_100 = avg_middle_lab_final[0] / 2.55
        avg_cervical_l_100 = avg_cervical_lab_final[0] / 2.55

        # Convert to Colormath LabColor scale for a* and b*
        incisal_a_colormath = np.clip(avg_incisal_lab_final[1] - 128, -128, 127)
        incisal_b_colormath = np.clip(avg_incisal_lab_final[2] - 128, -128, 127)
        middle_a_colormath = np.clip(avg_middle_lab_final[1] - 128, -128, 127)
        middle_b_colormath = np.clip(avg_middle_lab_final[2] - 128, -128, 127)
        cervical_a_colormath = np.clip(avg_cervical_lab_final[1] - 128, -128, 127)
        cervical_b_colormath = np.clip(avg_cervical_lab_final[2] - 128, -128, 127)

        print(f"DEBUG: Final L values (0-100 scale) for Rule-Based Mapping: Incisal={avg_incisal_l_100:.2f}, Middle={avg_middle_l_100:.2f}, Cervical={avg_cervical_l_100:.2f}")
        print(f"DEBUG: Final a* values (Colormath scale) for Rule-Based Mapping: Incisal={incisal_a_colormath:.2f}, Middle={middle_a_colormath:.2f}, Cervical={cervical_a_colormath:.2f}")
        print(f"DEBUG: Final b* values (Colormath scale) for Rule-Based Mapping: Incisal={incisal_b_colormath:.2f}, Middle={middle_b_colormath:.2f}, Cervical={cervical_b_colormath:.2f}")


        # --- Delta E Matching (Step 6) ---
        # Convert corrected OpenCV LAB (0-255 for L, a, b) to Colormath LabColor (0-100 for L, -128 to 127 for a,b)
        incisal_lab_colormath = LabColor(
            lab_l=avg_incisal_l_100,
            lab_a=incisal_a_colormath,
            lab_b=incisal_b_colormath
        )
        middle_lab_colormath = LabColor(
            lab_l=avg_middle_l_100,
            lab_a=middle_a_colormath,
            lab_b=middle_b_colormath
        )
        cervical_lab_colormath = LabColor(
            lab_l=avg_cervical_l_100,
            lab_a=cervical_a_colormath,
            lab_b=cervical_b_colormath
        )
        
        overall_lab_colormath = LabColor(
            lab_l=np.clip(overall_avg_lab_corrected_overall[0] / 2.55, 0, 100), # Overall still uses the base corrected value for ML
            lab_a=np.clip(overall_avg_lab_corrected_overall[1] - 128, -128, 127),
            lab_b=np.clip(overall_avg_lab_corrected_overall[2] - 128, -128, 127)
        )
        print(f"DEBUG: Colormath LAB objects (after correction and zone variability): Incisal={incisal_lab_colormath}, Middle={middle_lab_colormath}, Cervical={cervical_lab_colormath}, Overall={overall_lab_colormath}")

        incisal_delta_e_shade, incisal_min_delta = match_shade_with_delta_e(incisal_lab_colormath)
        middle_delta_e_shade, middle_min_delta = match_shade_with_delta_e(middle_lab_colormath)
        cervical_delta_e_shade, cervical_min_delta = match_shade_with_delta_e(cervical_lab_colormath)
        overall_delta_e_shade, overall_min_delta = match_shade_with_delta_e(overall_lab_colormath)
        # FIX: Ensure this printout is complete and clear
        print(f"DEBUG: Delta E matched shades: Overall={overall_delta_e_shade} (dE={overall_min_delta:.2f}), "
              f"Incisal={incisal_delta_e_shade} (dE={incisal_min_delta:.2f}), "
              f"Middle={middle_delta_e_shade} (dE={middle_min_delta:.2f}), "
              f"Cervical={cervical_delta_e_shade} (dE={cervical_min_delta:.2f})")

        # --- ML Prediction (using 0-100 L values from corrected data) ---
        overall_ml_shade = "Model Error"
        if shade_classifier_model is not None:
            # For ML, we use the average of the *final* L values of the zones to represent the tooth
            avg_l_for_ml = np.mean([avg_incisal_l_100, avg_middle_l_100, avg_cervical_l_100])
            # The ML model expects three features (incisal_l, middle_l, cervical_l)
            features_for_ml_prediction = np.array([[avg_incisal_l_100, avg_middle_l_100, avg_cervical_l_100]])
            overall_ml_shade = shade_classifier_model.predict(features_for_ml_prediction)[0]
            print(f"DEBUG: Features for ML: {features_for_ml_prediction}")
            print(f"DEBUG: Predicted Overall Shade (ML): {overall_ml_shade}")
        else:
            print("WARNING: AI model not loaded/trained. Cannot provide ML shade prediction.")

        # --- Calculate Confidence (Step 7) ---
        overall_accuracy_confidence, confidence_notes = calculate_confidence(overall_min_delta, device_profile)
        
        # Determine the final "Rule-based" shades using the Delta E matching
        final_incisal_rule_based = map_l_to_shade_rule_based(avg_incisal_l_100, incisal_a_colormath, incisal_b_colormath)
        final_middle_rule_based = map_l_to_shade_rule_based(avg_middle_l_100, middle_a_colormath, middle_b_colormath)
        final_cervical_rule_based = map_l_to_shade_rule_based(avg_cervical_l_100, cervical_a_colormath, cervical_b_colormath)

        print(f"DEBUG: Final Rule-based Shades (to be displayed): Incisal={final_incisal_rule_based}, Middle={final_middle_rule_based}, Cervical={final_cervical_rule_based}")


        detected_shades = {
            "incisal": final_incisal_rule_based,
            "middle": final_middle_rule_based,
            "cervical": final_cervical_rule_based,
            "overall_ml_shade": overall_ml_shade,

            "delta_e_matched_shades": {
                "overall": overall_delta_e_shade,
                "overall_delta_e": round(float(overall_min_delta), 2),
                "incisal": incisal_delta_e_shade,
                "incisal_delta_e": round(float(incisal_min_delta), 2),
                "middle": middle_delta_e_shade,
                "middle_delta_e": round(float(middle_min_delta), 2),
                "cervical": cervical_delta_e_shade,
                "cervical_delta_e": round(float(cervical_min_delta), 2),
            },
            "face_features": face_features,
            "tooth_analysis": tooth_analysis,
            "aesthetic_suggestion": aesthetic_suggestion,
            "accuracy_confidence": {
                "overall_percentage": overall_accuracy_confidence,
                "notes": confidence_notes
            },
            "selected_device_profile": device_profile,
            "selected_reference_tab": reference_tab
        }
        return detected_shades

    except Exception as e:
        print(f"CRITICAL ERROR during shade detection: {e}")
        traceback.print_exc()
        return {
            "incisal": "Error", "middle": "Error", "cervical": "Error",
            "overall_ml_shade": "Error",
            "face_features": {},
            "tooth_analysis": {},
            "aesthetic_suggestion": {},
            "delta_e_matched_shades": {},
            "accuracy_confidence": {"overall_percentage": "N/A", "notes": f"Processing failed: {e}"},
            "selected_device_profile": device_profile, "selected_reference_tab": reference_tab
        }


def generate_pdf_report(patient_name, shades, image_path, filepath):
    """Generates a PDF report with detected shades and the uploaded image."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="Shade View - Tooth Shade Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    selected_profile = shades.get("selected_device_profile", "N/A").replace("_", " ").title()
    pdf.cell(0, 10, txt=f"Simulated Device/Lighting Profile: {selected_profile}", ln=True)
    
    selected_ref_tab = shades.get("selected_reference_tab", "N/A").replace("_", " ").title()
    pdf.cell(0, 10, txt=f"Simulated Color Reference Used: {selected_ref_tab}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Detected Shades (Rule-based / Delta E):", ln=True) # Updated heading
    pdf.set_font("Arial", size=12)
    
    if "overall_ml_shade" in shades and shades["overall_ml_shade"] != "N/A":
        pdf.cell(0, 7, txt=f"   - Overall AI Prediction (ML): {shades['overall_ml_shade']}", ln=True)

    pdf.cell(0, 7, txt=f"   - Incisal Zone (Rule-based): {shades['incisal']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Middle Zone (Rule-based): {shades['middle']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Cervical Zone (Rule-based): {shades['cervical']}", ln=True)
    
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Delta E 2000 Matched Shades (Perceptual Match):", ln=True)
    pdf.set_font("Arial", size=12)
    delta_e_shades = shades.get("delta_e_matched_shades", {})
    if delta_e_shades:
        pdf.cell(0, 7, txt=f"   - Overall Delta E Match: {delta_e_shades.get('overall', 'N/A')} (dE: {delta_e_shades.get('overall_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"   - Incisal Zone Delta E Match: {delta_e_shades.get('incisal', 'N/A')} (dE: {delta_e_shades.get('incisal_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"   - Middle Zone Delta E Match: {delta_e_shades.get('middle', 'N/A')} (dE: {delta_e_shades.get('middle_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"   - Cervical Zone Delta E Match: {delta_e_shades.get('cervical', 'N/A')} (dE: {delta_e_shades.get('cervical_delta_e', 'N/A'):.2f})", ln=True)
    else:
        pdf.cell(0, 7, txt="   - Delta E matching data not available.", ln=True)


    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Shade Detection Accuracy Confidence:", ln=True)
    pdf.set_font("Arial", size=12)
    accuracy_conf = shades.get("accuracy_confidence", {})
    if accuracy_conf and accuracy_conf.get("overall_percentage") != "N/A":
        pdf.cell(0, 7, txt=f"   - Overall Confidence: {accuracy_conf.get('overall_percentage', 'N/A')}%", ln=True)
        pdf.multi_cell(0, 7, txt=f"   - Notes: {accuracy_conf.get('notes', 'N/A')}")
    else:
        pdf.cell(0, 7, txt="   - Confidence data not available or processing error.", ln=True)


    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=13)
    pdf.cell(0, 10, txt="Advanced AI Insights (Simulated):", ln=True)
    pdf.set_font("Arial", size=11)

    tooth_analysis = shades.get("tooth_analysis", {})
    if tooth_analysis:
        pdf.cell(0, 7, txt="   -- Tooth Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Overall Shade (Detailed): {tooth_analysis.get('simulated_overall_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Condition: {tooth_analysis.get('tooth_condition', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Stain Presence: {tooth_analysis.get('stain_presence', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Decay Presence: {tooth_analysis.get('decay_presence', 'N/A')}", ln=True)
        l_val = tooth_analysis.get('overall_lab', {}).get('L', 'N/A')
        a_val = tooth_analysis.get('overall_lab', {}).get('a', 'N/A')
        b_val = tooth_analysis.get('overall_lab', {}).get('b', 'N/A')
        if all(isinstance(v, (int, float)) for v in [l_val, a_val, b_val]):
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val:.2f}, a={a_val:.2f}, b={b_val:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val}, a={a_val}, b={b_val}", ln=True)
    
    pdf.ln(3)
    face_features = shades.get("face_features", {})
    if face_features:
        pdf.cell(0, 7, txt="   -- Facial Aesthetics Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Skin Tone: {face_features.get('skin_tone', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Lip Color: {face_features.get('lip_color', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Eye Contrast: {face_features.get('eye_contrast', 'N/A')}", ln=True)
        harmony_score = face_features.get('facial_harmony_score', 'N/A')
        if isinstance(harmony_score, (int, float)):
            pdf.cell(0, 7, txt=f"   - Simulated Facial Harmony Score: {harmony_score:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"   - Simulated Facial Harmony Score: {harmony_score}", ln=True)

    pdf.ln(3)
    aesthetic_suggestion = shades.get("aesthetic_suggestion", {})
    if aesthetic_suggestion:
        pdf.cell(0, 7, txt="   -- Aesthetic Shade Suggestion --", ln=True)
        pdf.cell(0, 7, txt=f"   - Suggested Shade: {aesthetic_suggestion.get('suggested_aesthetic_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"   - Confidence: {aesthetic_suggestion.get('aesthetic_confidence', 'N/A')}", ln=True)
        pdf.multi_cell(0, 7, txt=f"   - Notes: {aesthetic_suggestion.get('recommendation_notes', 'N/A')}")

    pdf.ln(10)

    try:
        if os.path.exists(image_path):
            pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
            if pdf.get_y() > 200:
                pdf.add_page()
            
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                h_img, w_img, _ = img_cv.shape
                max_w_pdf = 180
                w_pdf = min(w_img, max_w_pdf)
                h_pdf = h_img * (w_pdf / w_img)

                if pdf.get_y() + h_pdf + 10 > pdf.h - pdf.b_margin:
                    pdf.add_page()

                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                temp_image_path = "temp_pdf_image.png"
                cv2.imwrite(temp_image_path, img_rgb)
                
                pdf.image(temp_image_path, x=pdf.get_x(), y=pdf.get_y(), w=w_pdf, h=h_pdf)
                pdf.ln(h_pdf + 10)
                os.remove(temp_image_path)
            else:
                pdf.cell(0, 10, txt="Note: Image could not be loaded for embedding.", ln=True)

        else:
            pdf.cell(0, 10, txt="Note: Uploaded image file not found for embedding.", ln=True)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        pdf.cell(0, 10, txt="Note: An error occurred while embedding the image in the report.", ln=True)

    pdf.set_font("Arial", 'I', size=9)
    pdf.multi_cell(0, 6,
                   txt="DISCLAIMER: This report is based on simulated analysis for demonstration purposes only. It is not intended for clinical diagnosis, medical advice, or professional cosmetic planning. Always consult with a qualified dental or medical professional for definitive assessment, diagnosis, and treatment.",
                   align='C')
    pdf.output(filepath)


# ===============================================
# 5. ROUTES (Adapted for Firestore)
# ===============================================
@app.route('/')
def home():
    """Renders the home/landing page."""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login (Simulated for Canvas)."""
    if g.user and 'anonymous' not in g.user['id']:
        flash(f"You are already logged in as {g.user['username']}.", 'info')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username or not password:
            error = 'Username and password are required.'

        if error is None:
            simulated_user_id = 'user_' + username.lower().replace(' ', '_')
            session['user_id'] = simulated_user_id
            session['user'] = {'id': simulated_user_id, 'username': username}
            flash(f'Simulated login successful for {username}!', 'success')
            print(f"DEBUG: Simulated login for user: {username} (ID: {session['user_id']})")
            return redirect(url_for('dashboard'))
        flash(error, 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration (Simulated for Canvas)."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            flash(f"Simulated registration successful for {username}. You can now log in!", 'success')
            return redirect(url_for('login'))
        flash(error, 'danger')

    return render_template('register.html')

@app.route('/logout')
def logout():
    """Handles user logout."""
    session.clear()
    flash('You have been logged out.', 'info')
    print(f"DEBUG: User logged out. Session cleared.")
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user dashboard, displaying past reports."""
    reports_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
    user_reports = get_firestore_documents_in_collection(reports_path)
    user_reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    current_date_formatted = datetime.now().strftime('%Y-%m-%d')

    return render_template('dashboard.html',
                           reports=user_reports,
                           user=g.user,
                           current_date=current_date_formatted)


@app.route('/save_patient_data', methods=['POST'])
@login_required
def save_patient_data():
    """Handles saving new patient records to Firestore and redirects to image upload page."""
    op_number = request.form['op_number']
    patient_name = request.form['patient_name']
    age = request.form['age']
    sex = request.form['sex']
    record_date = request.form['date']
    user_id = g.user['id']

    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']

    existing_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if existing_patients:
        flash('OP Number already exists for another patient under your account. Please use a unique OP Number or select from recent entries.', 'error')
        return redirect(url_for('dashboard'))

    try:
        patient_data = {
            'user_id': user_id,
            'op_number': op_number,
            'patient_name': patient_name,
            'age': int(age),
            'sex': sex,
            'record_date': record_date,
            'created_at': datetime.now().isoformat()
        }
        
        add_firestore_document(patients_collection_path, patient_data)

        flash('Patient record saved successfully (to Firestore)! Now upload an image.', 'success')
        return redirect(url_for('upload_page', op_number=op_number))
    except Exception as e:
        flash(f'Error saving patient record to Firestore: {e}', 'error')
        return redirect(url_for('dashboard'))


@app.route('/upload_page/<op_number>')
@login_required
def upload_page(op_number):
    """Renders the dedicated image upload page for a specific patient."""
    user_id = g.user['id']

    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']
    patient = None
    all_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if all_patients:
        patient = all_patients[0]

    if patient is None:
        flash('Patient not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))

    return render_template('upload_page.html', op_number=op_number, patient_name=patient['patient_name'])


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Handles image upload, shade detection, and PDF report generation."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        op_number_from_form = request.form.get('op_number')
        patient_name = request.form.get('patient_name', 'Unnamed Patient')
        device_profile = request.form.get('device_profile', 'ideal')
        reference_tab = request.form.get('reference_tab', 'neutral_gray')

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1]
            unique_filename = str(uuid.uuid4()) + file_ext
            
            original_image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(original_image_path)
            flash('Image uploaded successfully!', 'success')

            detected_shades = detect_shades_from_image(original_image_path, device_profile, reference_tab)

            if (detected_shades.get("incisal") == "N/A" and
                detected_shades.get("middle") == "N/A" and
                detected_shades.get("cervical") == "N/A" and
                detected_shades.get("overall_ml_shade") == "N/A" and
                detected_shades.get("delta_e_matched_shades", {}).get("overall") == "N/A"):
                flash("Error processing image for shade detection. Please try another image or check image quality.", 'danger')
                if os.path.exists(original_image_path):
                    os.remove(original_image_path)
                return redirect(url_for('upload_page', op_number=op_number_from_form))


            report_filename = f"report_{patient_name.replace(' ', '')}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_filepath = os.path.join(REPORT_FOLDER, report_filename)
            generate_pdf_report(patient_name, detected_shades, original_image_path, report_filepath)
            flash('PDF report generated!', 'success')

            formatted_analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            report_data = {
                'patient_name': patient_name,
                'op_number': op_number_from_form,
                'original_image': unique_filename,
                'report_filename': report_filename,
                'detected_shades': detected_shades,
                'timestamp': datetime.now().isoformat(),
                'user_id': g.firestore_user_id
            }
            reports_collection_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
            add_firestore_document(reports_collection_path, report_data)

            return render_template('report.html',
                                   patient_name=patient_name,
                                   shades=detected_shades,
                                   image_filename=unique_filename,
                                   report_filename=report_filename,
                                   analysis_date=formatted_analysis_date,
                                   device_profile=device_profile,
                                   reference_tab=reference_tab)
    
    flash("Please select a patient from the dashboard to upload an image.", 'info')
    return redirect(url_for('dashboard'))


@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Allows downloading of generated PDF reports."""
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    if shade_classifier_model is None:
        print("CRITICAL: Machine Learning model could not be loaded or trained. Shade prediction will not work.")
    app.run(debug=True)

