import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading

# --- Modern UI Color Palette ---
COLOR_PALETTE = {
    'BACKGROUND': '#0f0f0f',        # Deep black
    'SURFACE': '#1a1a1a',          # Dark gray
    'CARD': '#262626',             # Card background
    'PRIMARY': '#6366f1',          # Modern indigo
    'PRIMARY_LIGHT': '#8b5cf6',    # Purple accent
    'SUCCESS': '#10b981',          # Modern green
    'DANGER': '#ef4444',           # Modern red
    'WARNING': '#f59e0b',          # Modern orange
    'TEXT_PRIMARY': '#ffffff',      # White text
    'TEXT_SECONDARY': '#a1a1aa',    # Gray text
    'TEXT_MUTED': '#71717a',       # Muted text
    'ACCENT': '#06b6d4',           # Cyan accent
    'BORDER': '#3f3f46',           # Border color
}

# Configuration
CONFIG = {
    'model_path': r"C:\Users\piyan\Downloads\codes-all\deepfake3 - Copy\models\deepfake_model.h5", 
    'input_size': (128, 128),
    'threshold': 0.5,
    'history_file': 'detection_history.txt'
}

# --- Custom Objects for Keras Model Loading ---

class FixedDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=False)

CUSTOM_OBJECTS = {
    'swish': tf.nn.swish,
    'FixedDropout': FixedDropout
}

tf.keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)

class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or CONFIG['model_path']
        self.model = None
        self.history = []
        self.is_model_loaded = self.load_model()

    def load_model(self):
        """Load the deepfake detection model with custom objects"""
        try:
            with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
                self.model = load_model(self.model_path)
            print("✅ Model loaded successfully!")
            return True
        except FileNotFoundError:
            print(f"❌ Critical Error: Model file not found at: {self.model_path}")
            messagebox.showerror("Model Error", f"Model file not found at: {self.model_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            messagebox.showerror("Model Error", f"Error loading model with custom objects: {e}")
            return False

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}. Check file path and permissions.")

            original_height, original_width = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, CONFIG['input_size'])
            image_normalized = image_resized / 255.0
            image_batch = np.expand_dims(image_normalized, axis=0)

            return {
                'preprocessed': image_batch,
                'original': image_rgb,
                'dimensions': (original_width, original_height)
            }
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None

    def predict(self, image_path):
        """Run deepfake detection on an image"""
        if self.model is None or not self.is_model_loaded:
            print("❌ Model not loaded. Cannot run prediction.")
            return None

        start_time = time.time()
        processed_data = self.preprocess_image(image_path)
        
        if processed_data is None:
            return None

        try:
            prediction = self.model.predict(processed_data['preprocessed'], verbose=0)[0][0]
            confidence = float(prediction)
            
            fake_confidence = round(confidence * 100, 2)
            real_confidence = round((1 - confidence) * 100, 2)
            is_fake = confidence > CONFIG['threshold']
            processing_time = time.time() - start_time

            result = {
                'path': image_path,
                'filename': os.path.basename(image_path),
                'is_fake': is_fake,
                'fake_confidence': fake_confidence,
                'real_confidence': real_confidence,
                'raw_score': confidence,
                'dimensions': processed_data['dimensions'],
                'original_image': processed_data['original'],
                'processing_time': round(processing_time, 3)
            }

            self.history.append(result)
            self._save_to_history(result)

            return result
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def _save_to_history(self, result):
        """Save detection results to history file"""
        try:
            display_confidence = result['fake_confidence'] if result['is_fake'] else result['real_confidence']
            
            with open(CONFIG['history_file'], 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"File: {result['filename']} | "
                        f"RESULT: {'FAKE' if result['is_fake'] else 'REAL'} | "
                        f"Confidence: {display_confidence}% | "
                        f"Time: {result['processing_time']}s\n")
        except Exception as e:
            print(f"⚠ Could not save to history: {e}")

class ModernButton(tk.Button):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, style='primary', **kwargs):
        self.style = style
        
        # Define style colors
        styles = {
            'primary': {
                'bg': COLOR_PALETTE['PRIMARY'],
                'hover': COLOR_PALETTE['PRIMARY_LIGHT'],
                'fg': COLOR_PALETTE['TEXT_PRIMARY']
            },
            'success': {
                'bg': COLOR_PALETTE['SUCCESS'],
                'hover': '#059669',
                'fg': COLOR_PALETTE['TEXT_PRIMARY']
            },
            'danger': {
                'bg': COLOR_PALETTE['DANGER'],
                'hover': '#dc2626',
                'fg': COLOR_PALETTE['TEXT_PRIMARY']
            },
            'warning': {
                'bg': COLOR_PALETTE['WARNING'],
                'hover': '#d97706',
                'fg': COLOR_PALETTE['TEXT_PRIMARY']
            },
            'secondary': {
                'bg': COLOR_PALETTE['SURFACE'],
                'hover': COLOR_PALETTE['CARD'],
                'fg': COLOR_PALETTE['TEXT_PRIMARY']
            }
        }
        
        style_config = styles.get(style, styles['primary'])
        
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=style_config['bg'],
            fg=style_config['fg'],
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            **kwargs
        )
        
        self.default_bg = style_config['bg']
        self.hover_bg = style_config['hover']
        
        # Bind hover events
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        
    def _on_enter(self, event):
        self.configure(bg=self.hover_bg)
        
    def _on_leave(self, event):
        self.configure(bg=self.default_bg)

class ModernProgressBar:
    """Custom animated progress bar"""
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, height=6, bg=COLOR_PALETTE['BACKGROUND'], highlightthickness=0)
        self.progress = 0
        self.is_animating = False
        
        # Create rounded rectangle for progress bar
        self.bg_rect = self.canvas.create_rectangle(0, 1, 0, 5, fill=COLOR_PALETTE['BORDER'], outline='')
        self.progress_rect = self.canvas.create_rectangle(0, 1, 0, 5, fill=COLOR_PALETTE['PRIMARY'], outline='')
        
    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)
        self.canvas.bind('<Configure>', self._on_configure)
        
    def _on_configure(self, event):
        width = event.width
        self.canvas.coords(self.bg_rect, 0, 1, width, 5)
        self._update_progress()
        
    def _update_progress(self):
        if hasattr(self, 'canvas'):
            width = self.canvas.winfo_width()
            progress_width = (self.progress / 100) * width
            self.canvas.coords(self.progress_rect, 0, 1, progress_width, 5)
            
    def start_animation(self):
        self.is_animating = True
        self._animate()
        
    def stop_animation(self):
        self.is_animating = False
        self.progress = 0
        self._update_progress()
        
    def _animate(self):
        if self.is_animating:
            self.progress = (self.progress + 5) % 100
            self._update_progress()
            self.parent.after(50, self._animate)

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DEEP-GUARD | AI Deepfake Detection")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=COLOR_PALETTE['BACKGROUND'])
        
        # Set window icon and styling
        try:
            self.root.iconbitmap(default='')  # Remove default icon
        except:
            pass
        
        # Initialize detector
        self.detector = DeepfakeDetector()
        
        # Initialize variables
        self.current_image_tk = None
        self.analysis_thread = None
        
        # Status variables
        if not self.detector.is_model_loaded:
            self.status_text = "⚠️ Model loading failed - Check console for details"
        else:
            self.status_text = "🟢 System ready - Select an image to analyze"
        
        self.result_text = "Awaiting image analysis..."
        
        # Set up modern UI
        self._setup_modern_ui()
        
        # Add subtle entrance animation
        self._animate_entrance()
        
    def _setup_modern_ui(self):
        """Set up the modern dark UI"""
        
        # Main container with gradient effect
        main_container = tk.Frame(self.root, bg=COLOR_PALETTE['BACKGROUND'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header section with glassmorphism effect
        header_frame = tk.Frame(main_container, bg=COLOR_PALETTE['SURFACE'], height=120)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        # Title with modern typography
        title_container = tk.Frame(header_frame, bg=COLOR_PALETTE['SURFACE'])
        title_container.pack(expand=True)
        
        title_main = tk.Label(
            title_container,
            text="DEEP-GUARD",
            font=("Segoe UI", 32, "bold"),
            bg=COLOR_PALETTE['SURFACE'],
            fg=COLOR_PALETTE['TEXT_PRIMARY']
        )
        title_main.pack(pady=(20, 0))
        
        subtitle = tk.Label(
            title_container,
            text="AI-Powered Deepfake Detection System",
            font=("Segoe UI", 14),
            bg=COLOR_PALETTE['SURFACE'],
            fg=COLOR_PALETTE['TEXT_SECONDARY']
        )
        subtitle.pack(pady=(0, 20))
        
        # Content area
        content_frame = tk.Frame(main_container, bg=COLOR_PALETTE['BACKGROUND'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Image display
        left_panel = tk.Frame(content_frame, bg=COLOR_PALETTE['CARD'], width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image display area with modern styling
        image_header = tk.Frame(left_panel, bg=COLOR_PALETTE['CARD'], height=50)
        image_header.pack(fill=tk.X, padx=20, pady=(20, 0))
        
        tk.Label(
            image_header,
            text="📷 Image Analysis",
            font=("Segoe UI", 16, "bold"),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['TEXT_PRIMARY']
        ).pack(anchor='w')
        
        # Progress bar
        progress_frame = tk.Frame(left_panel, bg=COLOR_PALETTE['CARD'])
        progress_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
        
        self.progress_bar = ModernProgressBar(progress_frame)
        self.progress_bar.pack(fill=tk.X)
        
        # Image display
        self.image_frame = tk.Frame(left_panel, bg=COLOR_PALETTE['SURFACE'], relief='flat', bd=1)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="🖼️\n\nDrop an image here or click 'Select Image'\n\nSupported formats: JPG, PNG, BMP",
            font=("Segoe UI", 14),
            bg=COLOR_PALETTE['SURFACE'],
            fg=COLOR_PALETTE['TEXT_MUTED'],
            justify=tk.CENTER
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Right panel - Controls and results
        right_panel = tk.Frame(content_frame, bg=COLOR_PALETTE['CARD'], width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Controls section
        controls_header = tk.Frame(right_panel, bg=COLOR_PALETTE['CARD'])
        controls_header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(
            controls_header,
            text="🎛️ Controls",
            font=("Segoe UI", 16, "bold"),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['TEXT_PRIMARY']
        ).pack(anchor='w')
        
        # Control buttons
        controls_frame = tk.Frame(right_panel, bg=COLOR_PALETTE['CARD'])
        controls_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.select_btn = ModernButton(
            controls_frame,
            "📁 Select Image",
            self.select_and_analyze_image,
            style='primary',
            state=tk.NORMAL if self.detector.is_model_loaded else tk.DISABLED
        )
        self.select_btn.pack(fill=tk.X, pady=(0, 10))
        
        ModernButton(
            controls_frame,
            "🔄 Reset Analysis",
            self.reset_analysis,
            style='secondary'
        ).pack(fill=tk.X, pady=(0, 10))
        
        ModernButton(
            controls_frame,
            "📜 View History",
            self.show_history,
            style='warning'
        ).pack(fill=tk.X, pady=(0, 10))
        
        # Results section
        results_header = tk.Frame(right_panel, bg=COLOR_PALETTE['CARD'])
        results_header.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        tk.Label(
            results_header,
            text="📊 Analysis Results",
            font=("Segoe UI", 16, "bold"),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['TEXT_PRIMARY']
        ).pack(anchor='w')
        
        # Results display
        self.results_frame = tk.Frame(right_panel, bg=COLOR_PALETTE['SURFACE'])
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.result_label = tk.Label(
            self.results_frame,
            text=self.result_text,
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_PALETTE['SURFACE'],
            fg=COLOR_PALETTE['TEXT_SECONDARY'],
            wraplength=280,
            justify=tk.CENTER
        )
        self.result_label.pack(pady=20)
        
        self.status_label = tk.Label(
            self.results_frame,
            text=self.status_text,
            font=("Segoe UI", 10),
            bg=COLOR_PALETTE['SURFACE'],
            fg=COLOR_PALETTE['TEXT_MUTED'],
            wraplength=280,
            justify=tk.CENTER
        )
        self.status_label.pack(pady=(0, 20))
        
        # Bottom action bar
        bottom_frame = tk.Frame(main_container, bg=COLOR_PALETTE['SURFACE'], height=60)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)
        
        ModernButton(
            bottom_frame,
            "❌ Exit Application",
            self.root.destroy,
            style='danger'
        ).pack(side=tk.RIGHT, padx=20, pady=15)
        
        version_label = tk.Label(
            bottom_frame,
            text="v2.0 | Powered by TensorFlow & Modern UI",
            font=("Segoe UI", 9),
            bg=COLOR_PALETTE['SURFACE'],
            fg=COLOR_PALETTE['TEXT_MUTED']
        )
        version_label.pack(side=tk.LEFT, padx=20, pady=15)
        
    def _animate_entrance(self):
        """Subtle entrance animation"""
        self.root.attributes('-alpha', 0.0)
        self._fade_in()
        
    def _fade_in(self, alpha=0.0):
        """Fade in animation"""
        if alpha < 1.0:
            alpha += 0.1
            self.root.attributes('-alpha', alpha)
            self.root.after(30, lambda: self._fade_in(alpha))
            
    def select_and_analyze_image(self):
        """Select and analyze image with modern UI feedback"""
        if not self.detector.is_model_loaded:
            self._show_modern_error("Model Error", "The deepfake detection model failed to load. Please check the console for details.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select an Image for Analysis",
            filetypes=[
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("BMP Files", "*.bmp"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self._analyze_image_async, args=(file_path,))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
    def _analyze_image_async(self, file_path):
        """Analyze image asynchronously"""
        try:
            # Update UI on main thread
            self.root.after(0, lambda: self._update_analysis_ui("start", file_path))
            
            # Perform analysis
            result = self.detector.predict(file_path)
            
            # Update UI with results on main thread
            self.root.after(0, lambda: self._update_analysis_ui("complete", file_path, result))
            
        except Exception as e:
            self.root.after(0, lambda: self._update_analysis_ui("error", file_path, str(e)))
            
    def _update_analysis_ui(self, stage, file_path, result_or_error=None):
        """Update UI during analysis stages"""
        filename = os.path.basename(file_path)
        
        if stage == "start":
            self.result_label.config(text=f"🔍 Analyzing {filename}...", fg=COLOR_PALETTE['ACCENT'])
            self.status_label.config(text="⚡ Processing image data...")
            self.select_btn.config(state=tk.DISABLED, text="🔄 Processing...")
            self.progress_bar.start_animation()
            
        elif stage == "complete":
            self.progress_bar.stop_animation()
            self.select_btn.config(state=tk.NORMAL, text="📁 Select Image")
            
            if result_or_error:
                self._display_modern_image(result_or_error['original_image'])
                
                if result_or_error['is_fake']:
                    result_text = f"🚨 FAKE DETECTED\n{result_or_error['fake_confidence']}% confidence"
                    result_color = COLOR_PALETTE['DANGER']
                    status_text = f"⚠️ Deepfake detected in {result_or_error['processing_time']}s"
                else:
                    result_text = f"✅ AUTHENTIC IMAGE\n{result_or_error['real_confidence']}% confidence"
                    result_color = COLOR_PALETTE['SUCCESS']
                    status_text = f"✨ Authentic image verified in {result_or_error['processing_time']}s"
                
                self.result_label.config(text=result_text, fg=result_color)
                self.status_label.config(text=status_text)
                
                # Show detailed results after a short delay
                self.root.after(1000, lambda: self._show_modern_results(result_or_error))
            else:
                self.result_label.config(text="❌ Analysis Failed", fg=COLOR_PALETTE['DANGER'])
                self.status_label.config(text="Failed to process image")
                
        elif stage == "error":
            self.progress_bar.stop_animation()
            self.select_btn.config(state=tk.NORMAL, text="📁 Select Image")
            self.result_label.config(text="💥 Error Occurred", fg=COLOR_PALETTE['DANGER'])
            self.status_label.config(text=f"Error: {result_or_error}")
            
    def _display_modern_image(self, image_np):
        """Display image with modern styling"""
        try:
            image = Image.fromarray(image_np)
            
            # Calculate display size
            self.image_frame.update_idletasks()
            display_width = self.image_frame.winfo_width() - 60
            display_height = self.image_frame.winfo_height() - 60
            
            if display_width <= 1 or display_height <= 1:
                display_width, display_height = 500, 400
            
            # Resize image maintaining aspect ratio
            image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Apply subtle shadow effect
            shadow = image.filter(ImageFilter.GaussianBlur(2))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="", compound=tk.CENTER)
            self.current_image_tk = photo
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            
    def _show_modern_results(self, result):
        """Show detailed results in modern popup"""
        results_window = tk.Toplevel(self.root)
        results_window.title("🔍 Detailed Analysis Results")
        results_window.geometry("800x600")
        results_window.configure(bg=COLOR_PALETTE['BACKGROUND'])
        results_window.transient(self.root)
        results_window.grab_set()
        
        # Center the window
        results_window.update_idletasks()
        x = (results_window.winfo_screenwidth() // 2) - (800 // 2)
        y = (results_window.winfo_screenheight() // 2) - (600 // 2)
        results_window.geometry(f"800x600+{x}+{y}")
        
        # Main container
        main_frame = tk.Frame(results_window, bg=COLOR_PALETTE['BACKGROUND'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=COLOR_PALETTE['CARD'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        result_status = "🚨 FAKE DETECTED" if result['is_fake'] else "✅ AUTHENTIC IMAGE"
        header_color = COLOR_PALETTE['DANGER'] if result['is_fake'] else COLOR_PALETTE['SUCCESS']
        
        tk.Label(
            header_frame,
            text=result_status,
            font=("Segoe UI", 24, "bold"),
            bg=COLOR_PALETTE['CARD'],
            fg=header_color,
            pady=20
        ).pack()
        
        # Info grid
        info_frame = tk.Frame(main_frame, bg=COLOR_PALETTE['SURFACE'])
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        info_data = [
            ("📁 Filename:", result['filename']),
            ("📐 Dimensions:", f"{result['dimensions'][0]} × {result['dimensions'][1]} pixels"),
            ("⏱️ Processing Time:", f"{result['processing_time']} seconds"),
            ("🎯 Confidence:", f"{result['fake_confidence'] if result['is_fake'] else result['real_confidence']}%")
        ]
        
        for i, (label, value) in enumerate(info_data):
            row_frame = tk.Frame(info_frame, bg=COLOR_PALETTE['SURFACE'])
            row_frame.pack(fill=tk.X, padx=20, pady=10)
            
            tk.Label(
                row_frame,
                text=label,
                font=("Segoe UI", 12, "bold"),
                bg=COLOR_PALETTE['SURFACE'],
                fg=COLOR_PALETTE['TEXT_SECONDARY']
            ).pack(side=tk.LEFT)
            
            tk.Label(
                row_frame,
                text=value,
                font=("Segoe UI", 12),
                bg=COLOR_PALETTE['SURFACE'],
                fg=COLOR_PALETTE['TEXT_PRIMARY']
            ).pack(side=tk.RIGHT)
            
        # Confidence chart
        self._create_modern_chart(main_frame, result)
        
        # Close button
        ModernButton(
            main_frame,
            "✖️ Close Results",
            results_window.destroy,
            style='secondary'
        ).pack(pady=20)
        
    def _create_modern_chart(self, parent, result):
        """Create modern confidence chart"""
        fig = plt.Figure(figsize=(7, 4), dpi=100)
        fig.patch.set_facecolor(COLOR_PALETTE['BACKGROUND'])
        
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLOR_PALETTE['SURFACE'])
        
        # Data
        categories = ['Fake Detection', 'Real Detection']
        values = [result['fake_confidence'], result['real_confidence']]
        colors = [COLOR_PALETTE['DANGER'], COLOR_PALETTE['SUCCESS']]
        
        # Create bars
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Styling
        ax.set_ylim(0, 100)
        ax.set_ylabel('Confidence Level (%)', color=COLOR_PALETTE['TEXT_PRIMARY'], fontsize=12)
        ax.set_title('Detection Confidence Analysis', color=COLOR_PALETTE['TEXT_PRIMARY'], fontsize=14, fontweight='bold', pad=20)
        
        # Remove spines and customize
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLOR_PALETTE['TEXT_MUTED'])
        ax.spines['bottom'].set_color(COLOR_PALETTE['TEXT_MUTED'])
        
        # Customize ticks
        ax.tick_params(axis='x', colors=COLOR_PALETTE['TEXT_PRIMARY'], labelsize=11)
        ax.tick_params(axis='y', colors=COLOR_PALETTE['TEXT_PRIMARY'], labelsize=10)
        ax.grid(axis='y', alpha=0.3, color=COLOR_PALETTE['TEXT_MUTED'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value}%', ha='center', va='bottom',
                   color=COLOR_PALETTE['TEXT_PRIMARY'], fontweight='bold', fontsize=11)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def reset_analysis(self):
        """Reset analysis with smooth transition"""
        self.image_label.config(
            image='',
            text="🖼️\n\nDrop an image here or click 'Select Image'\n\nSupported formats: JPG, PNG, BMP",
            fg=COLOR_PALETTE['TEXT_MUTED']
        )
        self.current_image_tk = None
        self.result_label.config(text="Awaiting image analysis...", fg=COLOR_PALETTE['TEXT_SECONDARY'])
        self.status_label.config(text="🟢 System ready - Select an image to analyze")
        self.progress_bar.stop_animation()
        self.select_btn.config(state=tk.NORMAL, text="📁 Select Image")

    def show_history(self):
        """Show detection history in modern interface"""
        history_path = CONFIG['history_file']
        if not os.path.exists(history_path) or os.stat(history_path).st_size == 0:
            self._show_modern_info("📜 Detection History", "No detection history available yet.\n\nStart analyzing images to build your history!")
            return

        try:
            with open(history_path, 'r') as f:
                history_data = f.readlines()

            history_window = tk.Toplevel(self.root)
            history_window.title("📜 Detection History")
            history_window.geometry("1000x600")
            history_window.configure(bg=COLOR_PALETTE['BACKGROUND'])
            history_window.transient(self.root)

            # Center window
            history_window.update_idletasks()
            x = (history_window.winfo_screenwidth() // 2) - (1000 // 2)
            y = (history_window.winfo_screenheight() // 2) - (600 // 2)
            history_window.geometry(f"1000x600+{x}+{y}")

            # Main container
            main_frame = tk.Frame(history_window, bg=COLOR_PALETTE['BACKGROUND'])
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # Header
            header_frame = tk.Frame(main_frame, bg=COLOR_PALETTE['CARD'])
            header_frame.pack(fill=tk.X, pady=(0, 20))

            tk.Label(
                header_frame,
                text="📜 Detection History",
                font=("Segoe UI", 20, "bold"),
                bg=COLOR_PALETTE['CARD'],
                fg=COLOR_PALETTE['TEXT_PRIMARY'],
                pady=15
            ).pack()

            # History display
            text_frame = tk.Frame(main_frame, bg=COLOR_PALETTE['SURFACE'])
            text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

            # Scrollbar
            scrollbar = tk.Scrollbar(text_frame, bg=COLOR_PALETTE['CARD'], troughcolor=COLOR_PALETTE['SURFACE'])
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Text widget
            history_text = tk.Text(
                text_frame,
                yscrollcommand=scrollbar.set,
                font=("Consolas", 11),
                bg=COLOR_PALETTE['SURFACE'],
                fg=COLOR_PALETTE['TEXT_PRIMARY'],
                insertbackground=COLOR_PALETTE['PRIMARY'],
                selectbackground=COLOR_PALETTE['PRIMARY'],
                wrap=tk.WORD,
                padx=20,
                pady=20,
                relief='flat',
                bd=0
            )
            history_text.pack(fill=tk.BOTH, expand=True)
            scrollbar.config(command=history_text.yview)

            # Configure tags for colored text
            history_text.tag_configure("fake", foreground=COLOR_PALETTE['DANGER'], font=("Consolas", 11, "bold"))
            history_text.tag_configure("real", foreground=COLOR_PALETTE['SUCCESS'], font=("Consolas", 11, "bold"))
            history_text.tag_configure("header", foreground=COLOR_PALETTE['ACCENT'], font=("Consolas", 11, "bold"))

            # Insert history data with formatting
            for i, line in enumerate(history_data):
                start_index = history_text.index(tk.INSERT)
                history_text.insert(tk.END, line)
                end_index = history_text.index(tk.INSERT)

                # Apply color formatting
                if "FAKE" in line:
                    fake_start = line.find("FAKE")
                    fake_pos = f"{start_index}+{fake_start}c"
                    fake_end = f"{fake_pos}+4c"
                    history_text.tag_add("fake", fake_pos, fake_end)
                elif "REAL" in line:
                    real_start = line.find("REAL")
                    real_pos = f"{start_index}+{real_start}c"
                    real_end = f"{real_pos}+4c"
                    history_text.tag_add("real", real_pos, real_end)

            history_text.config(state=tk.DISABLED)

            # Control buttons
            button_frame = tk.Frame(main_frame, bg=COLOR_PALETTE['BACKGROUND'])
            button_frame.pack(fill=tk.X)

            ModernButton(
                button_frame,
                "🗑️ Clear History",
                lambda: self._clear_history(history_window),
                style='danger'
            ).pack(side=tk.LEFT)

            ModernButton(
                button_frame,
                "✖️ Close",
                history_window.destroy,
                style='secondary'
            ).pack(side=tk.RIGHT)

        except Exception as e:
            self._show_modern_error("History Error", f"Failed to display history:\n{str(e)}")

    def _clear_history(self, parent_window):
        """Clear detection history with confirmation"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all detection history?\n\nThis action cannot be undone.", parent=parent_window):
            try:
                open(CONFIG['history_file'], 'w').close()
                self.detector.history = []
                parent_window.destroy()
                self._show_modern_info("History Cleared", "Detection history has been successfully cleared.")
            except Exception as e:
                self._show_modern_error("Error", f"Failed to clear history:\n{str(e)}")

    def _show_modern_error(self, title, message):
        """Show modern styled error dialog"""
        error_window = tk.Toplevel(self.root)
        error_window.title(f"❌ {title}")
        error_window.geometry("400x200")
        error_window.configure(bg=COLOR_PALETTE['BACKGROUND'])
        error_window.transient(self.root)
        error_window.grab_set()

        # Center window
        error_window.update_idletasks()
        x = (error_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (error_window.winfo_screenheight() // 2) - (200 // 2)
        error_window.geometry(f"400x200+{x}+{y}")

        # Main frame
        main_frame = tk.Frame(error_window, bg=COLOR_PALETTE['CARD'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Error icon and title
        tk.Label(
            main_frame,
            text="❌",
            font=("Segoe UI", 32),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['DANGER']
        ).pack(pady=(10, 0))

        tk.Label(
            main_frame,
            text=title,
            font=("Segoe UI", 14, "bold"),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['TEXT_PRIMARY']
        ).pack(pady=(0, 10))

        # Message
        tk.Label(
            main_frame,
            text=message,
            font=("Segoe UI", 10),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['TEXT_SECONDARY'],
            wraplength=350,
            justify=tk.CENTER
        ).pack(pady=(0, 20))

        # Close button
        ModernButton(
            main_frame,
            "Close",
            error_window.destroy,
            style='danger'
        ).pack()

    def _show_modern_info(self, title, message):
        """Show modern styled info dialog"""
        info_window = tk.Toplevel(self.root)
        info_window.title(f"ℹ️ {title}")
        info_window.geometry("400x200")
        info_window.configure(bg=COLOR_PALETTE['BACKGROUND'])
        info_window.transient(self.root)
        info_window.grab_set()

        # Center window
        info_window.update_idletasks()
        x = (info_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (info_window.winfo_screenheight() // 2) - (200 // 2)
        info_window.geometry(f"400x200+{x}+{y}")

        # Main frame
        main_frame = tk.Frame(info_window, bg=COLOR_PALETTE['CARD'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Info icon and title
        tk.Label(
            main_frame,
            text="ℹ️",
            font=("Segoe UI", 32),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['PRIMARY']
        ).pack(pady=(10, 0))

        tk.Label(
            main_frame,
            text=title,
            font=("Segoe UI", 14, "bold"),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['TEXT_PRIMARY']
        ).pack(pady=(0, 10))

        # Message
        tk.Label(
            main_frame,
            text=message,
            font=("Segoe UI", 10),
            bg=COLOR_PALETTE['CARD'],
            fg=COLOR_PALETTE['TEXT_SECONDARY'],
            wraplength=350,
            justify=tk.CENTER
        ).pack(pady=(0, 20))

        # Close button
        ModernButton(
            main_frame,
            "Got it",
            info_window.destroy,
            style='primary'
        ).pack()

# Main entry point
if __name__ == "__main__":
    try:
        # Set DPI awareness for Windows
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # Create and run the application
    root = tk.Tk()
    
    try:
        # Apply modern window styling
        try:
            root.attributes('-transparentcolor', '')  # Enable transparency support
        except:
            pass
        
        app = DeepfakeDetectorApp(root)
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Application critical error: {e}")
        
        # Create emergency error window
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror(
            "Critical Application Error", 
            f"DEEP-GUARD failed to start:\n\n{str(e)}\n\nPlease check the console for more details."
        )
        error_root.destroy()