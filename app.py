import os
import json
import time
import uuid
import requests
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# --- Initialize Flask Application ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'a-very-secure-and-random-secret-key' # Change this in production

# --- Define Constants ---
MODEL_PATH = r"C:\Users\piyan\Downloads\codes-all\deepfake3 - Copy\models\deepfake_model.h5"
HISTORY_PATH = 'history.json'
FORUM_PATH = 'forum.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_WIDTH, IMG_HEIGHT = 128, 128

# --- API Configurations ---
try:
    # Configure Gemini API
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("⚠️ Warning: GEMINI_API_KEY environment variable not found.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    gemini_model = None

# --- Ensure Required Directories Exist ---
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Helper Functions ---

def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)

def load_forum_posts():
    if not os.path.exists(FORUM_PATH):
        return []
    try:
        with open(FORUM_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_forum_posts(posts):
    with open(FORUM_PATH, 'w') as f:
        json.dump(posts, f, indent=4)

def load_keras_model():
    try:
        model = load_model(MODEL_PATH)
        print("✅ AI Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Critical Error: Could not load AI model. {e}")
        return None

model = load_keras_model()

def get_gemini_explanation(result, confidence, image_path):
    if not gemini_model:
        return "Gemini API is not configured, so no detailed explanation can be provided."
    try:
        img = Image.open(image_path)
        if result == 'Real':
            prompt = f"My deepfake detection model determined this image is 'Real' with {confidence:.2f}% confidence. Please provide a detailed explanation of why this image appears to be authentic. Analyze its lighting, shadows, textures, and consistency to support this conclusion."
        else:
            prompt = f"My deepfake detection model determined this image is 'Fake' with {confidence:.2f}% confidence. Please provide a detailed explanation of why this image shows signs of being AI-generated or manipulated. Point out specific visual artifacts, inconsistencies, or unnatural features you can find in the image to support this conclusion."
        response = gemini_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        return "An error occurred while generating the detailed explanation."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# ==============================================================================
# --- Flask Routes ---
# ==============================================================================

@app.route('/')
def home():
    articles = []
    news_api_key = os.environ.get("NEWS_API_KEY")
    if news_api_key:
        query = 'deepfake OR "synthetic media"'
        url = (f'https://newsapi.org/v2/everything?q={query}&sortBy=popularity&language=en&pageSize=5&apiKey={news_api_key}')
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            articles = [art for art in data.get("articles", []) if art.get('urlToImage')]
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching news for homepage: {e}")
    return render_template('home.html', articles=articles)

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/news')
def news():
    news_api_key = os.environ.get("NEWS_API_KEY")
    if not news_api_key:
        flash("News API key is not configured. Please contact the administrator.", "error")
        return render_template('news.html', articles=[])
    query = 'deepfake OR "synthetic media" OR "AI generated image" OR misinformation'
    url = (f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={news_api_key}')
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = [art for art in data.get("articles", []) if art.get('urlToImage') and art.get('description')]
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching news: {e}")
        flash("Could not retrieve the latest news. Please try again later.", "error")
        articles = []
    return render_template('news.html', articles=articles)

@app.route('/forum')
def forum():
    posts = load_forum_posts()
    sorted_posts = sorted(posts, key=lambda p: p['timestamp'], reverse=True)
    return render_template('forum.html', posts=sorted_posts)

@app.route('/forum/post', methods=['POST'])
def new_post():
    content = request.form.get('content')
    if content:
        posts = load_forum_posts()
        post = {'id': str(uuid.uuid4()), 'content': content, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'replies': []}
        posts.append(post)
        save_forum_posts(posts)
    return redirect(url_for('forum'))

@app.route('/forum/reply/<post_id>', methods=['POST'])
def new_reply(post_id):
    reply_content = request.form.get('reply_content')
    if reply_content:
        posts = load_forum_posts()
        for post in posts:
            if post['id'] == post_id:
                reply = {'id': str(uuid.uuid4()), 'content': reply_content, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}
                post['replies'].append(reply)
                break
        save_forum_posts(posts)
    return redirect(url_for('forum'))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash("The AI model is currently offline. Please contact support.", "error")
        return redirect(url_for('dashboard'))
    if 'file' not in request.files:
        flash("No file was included in the request.", "error")
        return redirect(url_for('dashboard'))
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash("Invalid file. Please choose a valid PNG, JPG, or JPEG image.", "error")
        return redirect(url_for('dashboard'))
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_array = preprocess_image(file_path)
        if img_array is None:
            flash("The image could not be processed. It may be corrupted.", "error")
            return redirect(url_for('dashboard'))
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        is_fake = confidence > 0.5
        result = "Fake" if is_fake else "Real"
        display_confidence = (confidence if is_fake else 1 - confidence) * 100
        gemini_explanation = get_gemini_explanation(result, display_confidence, file_path)
        history_entry = {
            'filename': filename,
            'result': result,
            'confidence': f"{display_confidence:.2f}%",
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'explanation': gemini_explanation
        }
        save_history(history_entry)
        return render_template(
            'result.html',
            result=result,
            confidence=f"{display_confidence:.2f}",
            image_path=filename,
            explanation=gemini_explanation
        )
    except Exception as e:
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('dashboard'))

@app.route('/history')
def history():
    history_data = load_history()
    return render_template('history.html', history=history_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/safety')
def safety():
    return render_template('safety.html')

if __name__ == '__main__':
    app.run(debug=True)