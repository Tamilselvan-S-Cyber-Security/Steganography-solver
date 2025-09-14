# Wolf Image Bullender - CTF Image Analysis Tool

A powerful Python tool for CTF (Capture The Flag) image analysis and manipulation using tkinter GUI.

## Features

- **Image Loading**: Support for multiple image formats (JPG, PNG, BMP, TIFF, GIF, WebP)
- **Color Manipulation**: Remove blue color channel, enhance colors
- **Hidden Data Detection**: Advanced image analysis to reveal hidden information
- **Side-by-side Comparison**: View original and processed images simultaneously
- **Save Functionality**: Export processed images in various formats

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python wolf_image_bullender.py
   ```

2. Load an image using the "Load Image" button
3. Use the processing tools:
   - **Remove Blue**: Removes the blue color channel
   - **Enhance Colors**: Increases color saturation and contrast
   - **Find Hidden Data**: Applies various filters to reveal hidden information
4. Save the processed image using "Save Image"

## CTF Use Cases

- **Steganography**: Detect hidden messages in images
- **Color Channel Analysis**: Analyze individual color channels
- **Image Forensics**: Examine images for hidden data
- **Visual Cryptography**: Manipulate images to reveal secrets

## Requirements

- Python 3.7+
- Pillow (PIL)
- OpenCV
- NumPy
- tkinter (usually included with Python)

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- GIF (.gif)
- WebP (.webp)
