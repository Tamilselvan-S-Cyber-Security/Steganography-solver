import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps
import cv2
import numpy as np
import os
import base64
import binascii
from pathlib import Path

class AdvancedImageAnalysis:
    def __init__(self, root):
        self.root = root
        self.root.title("Wolf Image Bullender - Advanced CTF Analysis")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2b2b2b')
        self.root.minsize(1200, 800)
        
        # Variables
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.zoom_factor = 1.0
        self.max_display_size = 800  # Maximum display size for images
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Wolf Image Bullender - Advanced CTF Analysis", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding=10)
        file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Reset", command=self.reset_image).pack(side=tk.LEFT, padx=(0, 5))
        
        # Image info
        self.info_label = ttk.Label(file_frame, text="No image loaded")
        self.info_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Processing frame
        process_frame = ttk.LabelFrame(control_frame, text="CTF Analysis Tools", padding=10)
        process_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Analysis buttons row 1
        button_frame1 = ttk.Frame(process_frame)
        button_frame1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(button_frame1, text="Remove Blue", command=self.remove_blue).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame1, text="Enhance", command=self.enhance_colors).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame1, text="Find Hidden", command=self.find_hidden_data).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame1, text="LSB Analysis", command=self.lsb_analysis).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame1, text="Extract Text", command=self.extract_text).pack(side=tk.LEFT, padx=(0, 3))
        
        # Analysis buttons row 2
        button_frame2 = ttk.Frame(process_frame)
        button_frame2.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame2, text="Color Channels", command=self.analyze_color_channels).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame2, text="Histogram", command=self.show_histogram).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame2, text="Edge Detection", command=self.edge_detection).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame2, text="Noise Analysis", command=self.noise_analysis).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame2, text="Metadata", command=self.show_metadata).pack(side=tk.LEFT, padx=(0, 3))
        
        # Analysis buttons row 3
        button_frame3 = ttk.Frame(process_frame)
        button_frame3.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame3, text="Extract Text", command=self.extract_hidden_text).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame3, text="Binary Data", command=self.extract_binary_data).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame3, text="Hex Dump", command=self.show_hex_dump).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame3, text="ASCII Art", command=self.extract_ascii_art).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame3, text="Save Data", command=self.save_extracted_data).pack(side=tk.LEFT, padx=(0, 3))
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Images
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Zoom controls
        zoom_frame = ttk.LabelFrame(left_panel, text="Image Controls", padding=5)
        zoom_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="Fit to Screen", command=self.fit_to_screen).pack(side=tk.LEFT, padx=(0, 5))
        
        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Original image
        orig_frame = ttk.LabelFrame(left_panel, text="Original Image", padding=5)
        orig_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.original_canvas = tk.Canvas(orig_frame, bg='#1e1e1e', relief=tk.SUNKEN, bd=2)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Processed image
        proc_frame = ttk.LabelFrame(left_panel, text="Processed Image", padding=5)
        proc_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.processed_canvas = tk.Canvas(proc_frame, bg='#1e1e1e', relief=tk.SUNKEN, bd=2)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Analysis results
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Analysis results
        results_frame = ttk.LabelFrame(right_panel, text="Analysis Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, width=50, 
                                                    bg='#1e1e1e', fg='#ffffff', 
                                                    font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
    def log_result(self, message):
        """Add a message to the results text area"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = Image.open(file_path)
                self.current_image = self.original_image.copy()
                self.processed_image = None
                
                self.display_images()
                self.update_info()
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                self.log_result(f"Loaded image: {os.path.basename(file_path)}")
                self.log_result(f"Size: {self.original_image.size[0]}x{self.original_image.size[1]}")
                self.log_result(f"Mode: {self.original_image.mode}")
                self.log_result("-" * 50)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.processed_image.save(file_path)
                self.status_var.set(f"Saved: {os.path.basename(file_path)}")
                self.log_result(f"Saved processed image: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                
    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.processed_image = None
            self.display_images()
            self.status_var.set("Image reset to original")
            self.log_result("Image reset to original")
        else:
            messagebox.showwarning("Warning", "No image loaded")
            
    def display_images(self):
        # Clear canvases
        self.original_canvas.delete("all")
        self.processed_canvas.delete("all")
        
        if self.original_image is not None:
            # Display original image
            self.display_image_on_canvas(self.original_image, self.original_canvas)
            
            # Display current/processed image
            if self.processed_image is not None:
                self.display_image_on_canvas(self.processed_image, self.processed_canvas)
            else:
                self.display_image_on_canvas(self.current_image, self.processed_canvas)
                
    def display_image_on_canvas(self, image, canvas):
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet rendered, schedule for later
            self.root.after(100, lambda: self.display_image_on_canvas(image, canvas))
            return
            
        # Calculate scaling to fit canvas with zoom factor
        img_width, img_height = image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        base_scale = min(scale_x, scale_y, 1.0)  # Don't scale up beyond original
        
        # Apply zoom factor
        scale = base_scale * self.zoom_factor
        
        # Limit maximum size for performance
        max_size = self.max_display_size
        if img_width * scale > max_size or img_height * scale > max_size:
            scale = min(max_size / img_width, max_size / img_height, scale)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image with high quality
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Center image on canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Display image
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep a reference
        
        # Update zoom label
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        
    def zoom_in(self):
        """Increase zoom factor"""
        if self.current_image is not None:
            self.zoom_factor = min(self.zoom_factor * 1.25, 5.0)  # Max 500% zoom
            self.display_images()
            self.log_result(f"Zoom increased to {int(self.zoom_factor * 100)}%")
        
    def zoom_out(self):
        """Decrease zoom factor"""
        if self.current_image is not None:
            self.zoom_factor = max(self.zoom_factor / 1.25, 0.1)  # Min 10% zoom
            self.display_images()
            self.log_result(f"Zoom decreased to {int(self.zoom_factor * 100)}%")
            
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        if self.current_image is not None:
            self.display_images()
            self.log_result("Zoom reset to 100%")
            
    def fit_to_screen(self):
        """Fit image to screen size"""
        if self.current_image is not None:
            self.zoom_factor = 1.0
            self.display_images()
            self.log_result("Image fitted to screen")
        
    def remove_blue(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Remove blue channel (set to 0)
            img_array[:, :, 2] = 0  # Blue channel is index 2 in RGB
            
            # Convert back to PIL Image
            self.processed_image = Image.fromarray(img_array)
            self.display_images()
            self.status_var.set("Blue color removed")
            self.log_result("Blue color channel removed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove blue: {str(e)}")
            
    def enhance_colors(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            # Enhance color saturation
            enhancer = ImageEnhance.Color(self.current_image)
            enhanced = enhancer.enhance(2.0)  # Increase saturation
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            self.processed_image = contrast_enhancer.enhance(1.5)
            
            self.display_images()
            self.status_var.set("Colors enhanced")
            self.log_result("Colors enhanced (saturation + contrast)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to enhance colors: {str(e)}")
            
    def find_hidden_data(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Starting hidden data analysis...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Apply various filters to reveal hidden data
            # 1. Edge detection
            edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 50, 150)
            
            # 2. Histogram equalization
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            
            # 3. Apply filters
            filtered = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            # 4. Combine results
            result = cv2.addWeighted(equalized, 0.7, filtered, 0.3, 0)
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            
            self.processed_image = Image.fromarray(result_rgb)
            self.display_images()
            self.status_var.set("Hidden data analysis complete")
            self.log_result("Hidden data analysis complete")
            self.log_result("Applied: Edge detection, histogram equalization, Gaussian blur")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze hidden data: {str(e)}")
            
    def lsb_analysis(self):
        """Analyze Least Significant Bits for hidden data"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Starting LSB analysis...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Extract LSBs from each color channel
            r_lsb = img_array[:, :, 0] & 1
            g_lsb = img_array[:, :, 1] & 1
            b_lsb = img_array[:, :, 2] & 1
            
            # Create LSB image
            lsb_image = np.zeros_like(img_array)
            lsb_image[:, :, 0] = r_lsb * 255
            lsb_image[:, :, 1] = g_lsb * 255
            lsb_image[:, :, 2] = b_lsb * 255
            
            self.processed_image = Image.fromarray(lsb_image)
            self.display_images()
            self.status_var.set("LSB analysis complete")
            self.log_result("LSB analysis complete")
            self.log_result("Extracted least significant bits from all color channels")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform LSB analysis: {str(e)}")
            
    def extract_text(self):
        """Extract text from image using OCR-like techniques"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Starting text extraction...")
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
            
            self.processed_image = Image.fromarray(result_rgb)
            self.display_images()
            self.status_var.set("Text extraction complete")
            self.log_result("Text extraction complete")
            self.log_result("Applied: Grayscale conversion, Otsu thresholding, morphological operations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract text: {str(e)}")
            
    def update_info(self):
        if self.original_image is not None:
            width, height = self.original_image.size
            mode = self.original_image.mode
            self.info_label.config(text=f"Size: {width}x{height} | Mode: {mode}")
        else:
            self.info_label.config(text="No image loaded")
            
    def analyze_color_channels(self):
        """Analyze individual color channels"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Starting color channel analysis...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Create separate channel images
            r_channel = img_array.copy()
            r_channel[:, :, 1] = 0  # Remove green
            r_channel[:, :, 2] = 0  # Remove blue
            
            g_channel = img_array.copy()
            g_channel[:, :, 0] = 0  # Remove red
            g_channel[:, :, 2] = 0  # Remove blue
            
            b_channel = img_array.copy()
            b_channel[:, :, 0] = 0  # Remove red
            b_channel[:, :, 1] = 0  # Remove green
            
            # Combine channels side by side
            height, width = img_array.shape[:2]
            combined = np.zeros((height, width * 3, 3), dtype=np.uint8)
            combined[:, :width] = r_channel
            combined[:, width:width*2] = g_channel
            combined[:, width*2:] = b_channel
            
            self.processed_image = Image.fromarray(combined)
            self.display_images()
            self.status_var.set("Color channel analysis complete")
            self.log_result("Color channel analysis complete")
            self.log_result("Red | Green | Blue channels displayed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze color channels: {str(e)}")
            
    def show_histogram(self):
        """Display image histogram"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Generating histogram...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Calculate histograms for each channel
            r_hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([img_array], [1], None, [256], [0, 256])
            b_hist = cv2.calcHist([img_array], [2], None, [256], [0, 256])
            
            # Create histogram visualization
            hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
            
            # Normalize histograms
            r_hist = cv2.normalize(r_hist, r_hist, 0, 300, cv2.NORM_MINMAX)
            g_hist = cv2.normalize(g_hist, g_hist, 0, 300, cv2.NORM_MINMAX)
            b_hist = cv2.normalize(b_hist, b_hist, 0, 300, cv2.NORM_MINMAX)
            
            # Draw histograms
            for i in range(1, 256):
                cv2.line(hist_img, (i-1, 300 - int(r_hist[i-1])), (i, 300 - int(r_hist[i])), (255, 0, 0), 1)
                cv2.line(hist_img, (i-1, 300 - int(g_hist[i-1])), (i, 300 - int(g_hist[i])), (0, 255, 0), 1)
                cv2.line(hist_img, (i-1, 300 - int(b_hist[i-1])), (i, 300 - int(b_hist[i])), (0, 0, 255), 1)
            
            # Resize to match original image
            target_height = min(300, self.current_image.height)
            hist_img = cv2.resize(hist_img, (self.current_image.width, target_height))
            
            self.processed_image = Image.fromarray(hist_img)
            self.display_images()
            self.status_var.set("Histogram generated")
            self.log_result("Histogram generated (Red | Green | Blue)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate histogram: {str(e)}")
            
    def edge_detection(self):
        """Apply edge detection filters"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Applying edge detection...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply different edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edges_sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Normalize and combine
            edges_sobelx = np.uint8(np.absolute(edges_sobelx))
            edges_sobely = np.uint8(np.absolute(edges_sobely))
            edges_laplacian = np.uint8(np.absolute(edges_laplacian))
            
            # Combine all edge detection results
            combined = cv2.addWeighted(edges_canny, 0.4, edges_sobelx, 0.2, 0)
            combined = cv2.addWeighted(combined, 0.8, edges_sobely, 0.2, 0)
            combined = cv2.addWeighted(combined, 0.9, edges_laplacian, 0.1, 0)
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
            
            self.processed_image = Image.fromarray(result_rgb)
            self.display_images()
            self.status_var.set("Edge detection complete")
            self.log_result("Edge detection complete")
            self.log_result("Applied: Canny, Sobel X/Y, Laplacian filters")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply edge detection: {str(e)}")
            
    def noise_analysis(self):
        """Analyze image noise patterns"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Starting noise analysis...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply noise reduction and compare
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Calculate noise by difference
            noise = cv2.absdiff(gray, denoised)
            
            # Enhance noise for visibility
            noise_enhanced = cv2.multiply(noise, 3)
            noise_enhanced = np.clip(noise_enhanced, 0, 255).astype(np.uint8)
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(noise_enhanced, cv2.COLOR_GRAY2RGB)
            
            self.processed_image = Image.fromarray(result_rgb)
            self.display_images()
            self.status_var.set("Noise analysis complete")
            self.log_result("Noise analysis complete")
            self.log_result("Noise patterns extracted and enhanced")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze noise: {str(e)}")
            
    def show_metadata(self):
        """Display image metadata"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Extracting image metadata...")
            
            # Get basic image info
            width, height = self.current_image.size
            mode = self.current_image.mode
            format_name = self.current_image.format
            
            # Get EXIF data if available
            exif_data = {}
            if hasattr(self.current_image, '_getexif') and self.current_image._getexif() is not None:
                exif = self.current_image._getexif()
                for tag_id, value in exif.items():
                    tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            
            # Display metadata
            self.log_result("=== IMAGE METADATA ===")
            self.log_result(f"Dimensions: {width} x {height}")
            self.log_result(f"Mode: {mode}")
            self.log_result(f"Format: {format_name}")
            self.log_result(f"File size: {os.path.getsize(self.image_path) if self.image_path else 'Unknown'} bytes")
            
            if exif_data:
                self.log_result("=== EXIF DATA ===")
                for key, value in list(exif_data.items())[:10]:  # Show first 10 items
                    self.log_result(f"{key}: {value}")
                if len(exif_data) > 10:
                    self.log_result(f"... and {len(exif_data) - 10} more EXIF entries")
            else:
                self.log_result("No EXIF data found")
                
            self.status_var.set("Metadata extracted")
            self.log_result("Metadata extraction complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract metadata: {str(e)}")
            
    def extract_hidden_text(self):
        """Extract hidden text from image using LSB steganography"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Extracting hidden text using LSB steganography...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Extract LSBs
            binary_text = ""
            delimiter = "1111111111111110"
            
            for row in range(img_array.shape[0]):
                for col in range(img_array.shape[1]):
                    for channel in range(3):  # RGB channels
                        # Extract LSB
                        lsb = img_array[row, col, channel] & 1
                        binary_text += str(lsb)
                        
                        # Check for delimiter
                        if len(binary_text) >= 16 and binary_text[-16:] == delimiter:
                            # Remove delimiter and convert to text
                            message_binary = binary_text[:-16]
                            extracted_text = self.binary_to_text(message_binary)
                            
                            self.log_result("=== HIDDEN TEXT EXTRACTED ===")
                            self.log_result(f"Text: {extracted_text}")
                            self.log_result(f"Length: {len(extracted_text)} characters")
                            self.log_result(f"Binary length: {len(message_binary)} bits")
                            
                            # Check if it looks like a flag
                            if extracted_text.startswith("WOLF{") or extracted_text.startswith("CTF{") or extracted_text.startswith("flag{"):
                                self.log_result("🎯 POTENTIAL FLAG DETECTED!")
                                self.log_result(f"Flag: {extracted_text}")
                            
                            self.status_var.set("Hidden text extracted")
                            return
            
            # If no delimiter found, try to extract readable text
            if len(binary_text) > 0:
                # Try different lengths
                for length in range(8, min(len(binary_text), 1000), 8):
                    try:
                        partial_text = self.binary_to_text(binary_text[:length])
                        if self.is_readable_text(partial_text):
                            self.log_result("=== PARTIAL TEXT EXTRACTED ===")
                            self.log_result(f"Text: {partial_text}")
                            self.log_result(f"Length: {len(partial_text)} characters")
                            break
                    except:
                        continue
            
            self.log_result("No hidden text found with standard LSB extraction")
            self.status_var.set("Text extraction complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract hidden text: {str(e)}")
            
    def binary_to_text(self, binary):
        """Convert binary string to text"""
        return ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))
        
    def is_readable_text(self, text):
        """Check if text contains mostly readable characters"""
        if len(text) < 4:
            return False
        readable_chars = sum(1 for c in text if c.isprintable() and c not in '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f')
        return readable_chars / len(text) > 0.7
        
    def extract_binary_data(self):
        """Extract binary data from image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Extracting binary data from image...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Extract LSBs from all channels
            binary_data = ""
            for row in range(img_array.shape[0]):
                for col in range(img_array.shape[1]):
                    for channel in range(3):  # RGB channels
                        lsb = img_array[row, col, channel] & 1
                        binary_data += str(lsb)
            
            self.log_result("=== BINARY DATA EXTRACTED ===")
            self.log_result(f"Total bits: {len(binary_data)}")
            self.log_result(f"Total bytes: {len(binary_data) // 8}")
            
            # Show first 200 bits
            preview = binary_data[:200]
            self.log_result(f"First 200 bits: {preview}")
            if len(binary_data) > 200:
                self.log_result("... (truncated)")
            
            # Try to find patterns
            self.log_result("=== PATTERN ANALYSIS ===")
            
            # Look for common delimiters
            delimiters = ["1111111111111110", "1111111111111111", "0000000000000000"]
            for delimiter in delimiters:
                if delimiter in binary_data:
                    pos = binary_data.find(delimiter)
                    self.log_result(f"Found delimiter '{delimiter}' at position {pos}")
                    if pos > 0:
                        message_bits = binary_data[:pos]
                        try:
                            message = self.binary_to_text(message_bits)
                            if self.is_readable_text(message):
                                self.log_result(f"Message before delimiter: {message}")
                        except:
                            pass
            
            # Look for repeated patterns
            patterns = ["01" * 10, "10" * 10, "0000", "1111"]
            for pattern in patterns:
                count = binary_data.count(pattern)
                if count > 5:
                    self.log_result(f"Pattern '{pattern}' found {count} times")
            
            self.status_var.set("Binary data extracted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract binary data: {str(e)}")
            
    def show_hex_dump(self):
        """Show hex dump of image data"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Generating hex dump of image data...")
            
            # Convert to numpy array
            img_array = np.array(self.current_image)
            
            # Flatten the array
            flat_data = img_array.flatten()
            
            self.log_result("=== HEX DUMP ===")
            self.log_result(f"Total bytes: {len(flat_data)}")
            
            # Show hex dump in 16-byte rows
            for i in range(0, min(len(flat_data), 512), 16):  # Show first 512 bytes
                hex_row = ' '.join(f'{b:02x}' for b in flat_data[i:i+16])
                ascii_row = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in flat_data[i:i+16])
                self.log_result(f"{i:08x}: {hex_row:<48} |{ascii_row}|")
            
            if len(flat_data) > 512:
                self.log_result("... (truncated)")
            
            # Look for interesting patterns
            self.log_result("=== PATTERN SEARCH ===")
            
            # Look for common file signatures
            signatures = {
                b'\x89PNG': 'PNG signature',
                b'\xff\xd8\xff': 'JPEG signature',
                b'GIF8': 'GIF signature',
                b'BM': 'BMP signature',
                b'RIFF': 'RIFF signature',
                b'PK': 'ZIP/PDF signature'
            }
            
            data_bytes = flat_data.tobytes()
            for sig, name in signatures.items():
                if sig in data_bytes:
                    pos = data_bytes.find(sig)
                    self.log_result(f"Found {name} at offset {pos:08x}")
            
            # Look for text strings
            text_strings = []
            current_string = ""
            for byte in flat_data[:1000]:  # Check first 1000 bytes
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                else:
                    if len(current_string) >= 4:
                        text_strings.append(current_string)
                    current_string = ""
            
            if text_strings:
                self.log_result("=== TEXT STRINGS FOUND ===")
                for string in text_strings[:10]:  # Show first 10 strings
                    self.log_result(f"'{string}'")
            
            self.status_var.set("Hex dump generated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate hex dump: {str(e)}")
            
    def extract_ascii_art(self):
        """Extract ASCII art from image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.log_result("Extracting ASCII art from image...")
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2GRAY)
            
            # Resize to reasonable size for ASCII
            height, width = gray.shape
            max_width = 80
            max_height = 40
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height))
            
            # ASCII characters from dark to light
            ascii_chars = " .:-=+*#%@"
            
            ascii_art = ""
            for row in gray:
                for pixel in row:
                    # Map pixel value (0-255) to ASCII character
                    char_index = int(pixel / 255 * (len(ascii_chars) - 1))
                    ascii_art += ascii_chars[char_index]
                ascii_art += "\n"
            
            self.log_result("=== ASCII ART ===")
            self.log_result(ascii_art)
            
            # Also try LSB-based ASCII art
            self.log_result("=== LSB ASCII ART ===")
            img_array = np.array(self.current_image)
            lsb_ascii = ""
            
            for row in range(min(img_array.shape[0], 40)):
                for col in range(min(img_array.shape[1], 80)):
                    # Extract LSB from all channels
                    lsb = (img_array[row, col, 0] & 1) + (img_array[row, col, 1] & 1) + (img_array[row, col, 2] & 1)
                    if lsb == 0:
                        lsb_ascii += " "
                    elif lsb == 1:
                        lsb_ascii += "."
                    elif lsb == 2:
                        lsb_ascii += "+"
                    else:
                        lsb_ascii += "#"
                lsb_ascii += "\n"
            
            self.log_result(lsb_ascii)
            
            self.status_var.set("ASCII art extracted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract ASCII art: {str(e)}")
            
    def save_extracted_data(self):
        """Save extracted data to files"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            from tkinter import filedialog
            
            # Ask for save directory
            save_dir = filedialog.askdirectory(title="Select directory to save extracted data")
            if not save_dir:
                return
            
            self.log_result(f"Saving extracted data to: {save_dir}")
            
            # Extract text
            img_array = np.array(self.current_image)
            binary_data = ""
            for row in range(img_array.shape[0]):
                for col in range(img_array.shape[1]):
                    for channel in range(3):
                        lsb = img_array[row, col, channel] & 1
                        binary_data += str(lsb)
            
            # Save binary data
            with open(f"{save_dir}/extracted_binary.txt", "w") as f:
                f.write(binary_data)
            
            # Save hex dump
            flat_data = img_array.flatten()
            with open(f"{save_dir}/hex_dump.txt", "w") as f:
                for i in range(0, len(flat_data), 16):
                    hex_row = ' '.join(f'{b:02x}' for b in flat_data[i:i+16])
                    ascii_row = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in flat_data[i:i+16])
                    f.write(f"{i:08x}: {hex_row:<48} |{ascii_row}|\n")
            
            # Save ASCII art
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            max_width, max_height = 80, 40
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height))
            
            ascii_chars = " .:-=+*#%@"
            ascii_art = ""
            for row in gray:
                for pixel in row:
                    char_index = int(pixel / 255 * (len(ascii_chars) - 1))
                    ascii_art += ascii_chars[char_index]
                ascii_art += "\n"
            
            with open(f"{save_dir}/ascii_art.txt", "w") as f:
                f.write(ascii_art)
            
            # Try to extract and save text
            delimiter = "1111111111111110"
            if delimiter in binary_data:
                pos = binary_data.find(delimiter)
                message_bits = binary_data[:pos]
                try:
                    extracted_text = self.binary_to_text(message_bits)
                    with open(f"{save_dir}/extracted_text.txt", "w") as f:
                        f.write(extracted_text)
                    self.log_result(f"Extracted text saved: {extracted_text}")
                except:
                    pass
            
            self.log_result("✅ All extracted data saved successfully!")
            self.log_result(f"Files saved in: {save_dir}")
            self.status_var.set("Data saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save extracted data: {str(e)}")

def main():
    root = tk.Tk()
    app = AdvancedImageAnalysis(root)
    root.mainloop()

if __name__ == "__main__":
    main()
