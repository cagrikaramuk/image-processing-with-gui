import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QFileDialog, QScrollArea, QInputDialog, QMessageBox, QGridLayout
)

# Assuming main.py is in the same directory
import main as img_ops 

# Helper function to convert function name to readable label
def format_button_text(name):
    name = name.replace("apply_", "").replace("get_", "").replace("calculate_", "")
    name = name.replace("_", " ").title()
    # Specific cosmetic adjustments for button labels
    if name == "Split Rgb Channels": name = "Split RGB Channels"
    if name == "Equalize Histogram": name = "Equalize Hist"
    if name == "Roberts Cross Edge Detection": name = "Roberts Cross"
    if name == "Laplacian Edge Detection": name = "Laplacian"
    if name == "Kmeans Segmentation": name = "K-Means Segment"
    if name == "Dft": name = "Calculate DFT"
    return name

class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 1000, 700) # Adjusted initial size

        # --- Application State ---
        self.current_image = None # Original loaded image (BGR)
        self.original_image_path = None # Path of the loaded image
        self.processed_image = None # Current image being worked on / displayed
        self.dft_shift = None # Store DFT result for spectrum display

        # --- Main UI Setup ---
        self._setup_ui()
        self.populate_function_buttons() # Populate dynamic buttons

    def _setup_ui(self):
        """Helper to initialize the main UI layout and widgets."""
        main_layout = QHBoxLayout()

        # Left side for image display
        image_display_layout = QVBoxLayout()
        self.image_label = QLabel("Load an image to start")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        
        image_display_layout.addWidget(scroll_area)
        main_layout.addLayout(image_display_layout, 3) # Give more space to image

        # Right side for controls
        controls_layout = QVBoxLayout()
        
        # File Buttons (Load/Save)
        file_button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False) 
        file_button_layout.addWidget(self.load_button)
        file_button_layout.addWidget(self.save_button)
        controls_layout.addLayout(file_button_layout)

        self.functions_group_box = QWidget() 
        self.functions_layout = QGridLayout() 
        self.functions_layout.setAlignment(Qt.AlignmentFlag.AlignTop) 
        self.functions_group_box.setLayout(self.functions_layout)
        
        functions_scroll_area = QScrollArea()
        functions_scroll_area.setWidgetResizable(True)
        functions_scroll_area.setWidget(self.functions_group_box)
        
        controls_layout.addWidget(functions_scroll_area)
        main_layout.addLayout(controls_layout, 1) # Less space for controls

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def populate_function_buttons(self):
        """Dynamically creates and adds function buttons to the layout."""
        # Clear existing buttons if any (e.g., if this were called multiple times)
        while self.functions_layout.count():
             item = self.functions_layout.takeAt(0)
             widget = item.widget()
             if widget is not None:
                 widget.deleteLater()

        # Define operations: simple ones (image -> image)
        simple_ops = [
            img_ops.convert_to_gray,
            img_ops.get_negative,
            img_ops.equalize_histogram,
            img_ops.apply_conservative_filter,
            img_ops.apply_prewitt_edge_detection,
            img_ops.apply_roberts_cross_edge_detection,
        ]

        # Operations requiring parameters (fetched via QInputDialog)
        # Format: func: [ (label, type_str, min, max, default, optional_decimals_or_step), ... ]
        param_ops = {
            img_ops.adjust_brightness: [("Brightness Value:", "int", -255, 255, 0)],
            img_ops.apply_threshold: [("Threshold Value:", "int", 0, 255, 127)],
            img_ops.adjust_contrast: [("Alpha (Contrast):", "double", 0.1, 3.0, 1.0, 1), ("Beta (Brightness):", "int", -100, 100, 0)],
            img_ops.translate_image: [("Translate X:", "int", -500, 500, 50), ("Translate Y:", "int", -500, 500, 50)],
            img_ops.flip_image: [("Flip Code (0=V, 1=H, -1=Both):", "int", -1, 1, 1)],
            img_ops.shear_image: [("Shear X Factor:", "double", -1.0, 1.0, 0.2, 2), ("Shear Y Factor:", "double", -1.0, 1.0, 0.0, 2)],
            img_ops.resize_image: [("Scale Factor:", "double", 0.1, 5.0, 0.5, 2)], # Simple scaling
            img_ops.rotate_image: [("Angle (degrees):", "double", -360.0, 360.0, 45.0, 1)],
            # Crop and Perspective Transform require more complex input than QInputDialog
            img_ops.apply_mean_filter: [("Kernel Size (odd):", "int", 3, 21, 3, 2)],
            img_ops.apply_median_filter: [("Kernel Size (odd):", "int", 3, 21, 3, 2)],
            img_ops.apply_gaussian_filter: [("Kernel Size (odd):", "int", 3, 21, 3, 2), ("Sigma (0=auto):", "double", 0.0, 10.0, 0.0, 1)],
            img_ops.apply_laplacian_edge_detection: [("Kernel Size (odd):", "int", 1, 7, 3, 2)],
            img_ops.apply_sobel_edge_detection: [("dx (0 or 1):", "int", 0, 1, 1), ("dy (0 or 1):", "int", 0, 1, 1), ("Kernel Size (odd):", "int", 1, 7, 3, 2)],
            img_ops.apply_canny_edge_detection: [("Threshold 1:", "int", 0, 500, 100), ("Threshold 2:", "int", 0, 500, 200)],
            img_ops.apply_erosion: [("Kernel Size:", "int", 1, 11, 3), ("Iterations:", "int", 1, 10, 1)],
            img_ops.apply_dilation: [("Kernel Size:", "int", 1, 11, 3), ("Iterations:", "int", 1, 10, 1)],
            img_ops.apply_kmeans_segmentation: [("Number of Clusters (k):", "int", 2, 10, 3)],
        }

        # Add buttons to the grid layout
        row, col = 0, 0
        max_cols = 2 # Number of columns for buttons

        # Helper to add a button and advance grid position
        def add_button_to_grid(button):
            nonlocal row, col
            button.setEnabled(False) # Initially disabled, enabled on image load
            self.functions_layout.addWidget(button, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        for func in simple_ops:
            btn = QPushButton(format_button_text(func.__name__))
            btn.clicked.connect(lambda checked=False, f=func: self.apply_operation_simple(f))
            add_button_to_grid(btn)

        for func, params_meta in param_ops.items():
            btn = QPushButton(format_button_text(func.__name__))
            btn.clicked.connect(lambda checked=False, f=func, p=params_meta: self.apply_operation_with_params(f, p))
            add_button_to_grid(btn)
                
        # --- Special Function Buttons ---
        dft_button = QPushButton(format_button_text(img_ops.calculate_dft.__name__))
        dft_button.clicked.connect(self.run_calculate_dft)
        add_button_to_grid(dft_button)
        
        self.spectrum_button = QPushButton("Show Mag Spectrum")
        self.spectrum_button.clicked.connect(self.show_magnitude_spectrum)
        self.spectrum_button.setEnabled(False) # Enabled only after DFT runs
        self.functions_layout.addWidget(self.spectrum_button, row, col); col += 1 # Manual add due to unique enable logic
        if col >= max_cols: col = 0; row += 1
        
        hist_button = QPushButton(format_button_text(img_ops.calculate_histogram.__name__))
        hist_button.clicked.connect(self.show_histogram)
        add_button_to_grid(hist_button)

        split_button = QPushButton(format_button_text(img_ops.split_rgb_channels.__name__))
        split_button.clicked.connect(self.show_split_channels)
        add_button_to_grid(split_button)

        # --- Reset Button ---
        reset_button = QPushButton("Reset to Original")
        reset_button.clicked.connect(self.reset_image)
        add_button_to_grid(reset_button)

    # --- File Operations ---
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)")
        if file_name:
            self.original_image_path = file_name
            try:
                self.current_image = img_ops.read_image_BGR(file_name)
                if self.current_image is None:
                    raise ValueError("Image data could not be read (None returned).")
                self.processed_image = self.current_image.copy()
                self.display_image(self.processed_image)
                self.save_button.setEnabled(True)
                self.dft_shift = None # Reset DFT cache
                self._set_operation_buttons_enabled(True)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load image from: {file_name}\nError: {e}")
                self.save_button.setEnabled(False)
                self._set_operation_buttons_enabled(False)


    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Save Error", "No processed image to save.")
            return

        suggested_name = "processed_image.png"
        save_dir = ""
        if self.original_image_path:
            p = Path(self.original_image_path)
            suggested_name = p.stem + "_processed" + p.suffix
            save_dir = str(p.parent)

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", 
                                                  str(Path(save_dir) / suggested_name), 
                                                  "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;Bitmap Files (*.bmp);;TIFF Files (*.tif)")
        if file_name:
            try:
                img_ops.save_image(self.processed_image, file_name)
                QMessageBox.information(self, "Save Success", f"Image saved to:\n{file_name}")
            except Exception as e:
                 QMessageBox.critical(self, "Save Error", f"Could not save image.\nError: {e}")

    # --- Image Display ---
    def display_image(self, cv_image):
        if cv_image is None:
            self.image_label.setText("No image to display.")
            self.image_label.setPixmap(QPixmap())
            return

        try:
            q_image = None
            if cv_image.ndim == 3: # Color image (assumed BGR)
                height, width, _ = cv_image.shape
                bytes_per_line = 3 * width
                contiguous_image = np.ascontiguousarray(cv_image)
                q_image = QImage(contiguous_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
            elif cv_image.ndim == 2: # Grayscale image
                height, width = cv_image.shape
                bytes_per_line = 1 * width
                contiguous_image = np.ascontiguousarray(cv_image)
                q_image = QImage(contiguous_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            if q_image:
                pixmap = QPixmap.fromImage(q_image)
                # Scale pixmap smoothly, keeping aspect ratio, fitting within the label
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                 QMessageBox.warning(self,"Display Error", "Cannot display: unsupported image format (not 2D or 3D).")
                 self.image_label.setPixmap(QPixmap())
        except Exception as e:
            print(f"Error displaying image: {e}") # For debugging
            QMessageBox.critical(self,"Display Error", f"An error occurred while displaying the image:\n{e}")
            self.image_label.setText("Error displaying image.")
            self.image_label.setPixmap(QPixmap())

    # --- Core Operation Handlers ---
    def apply_operation_simple(self, operation_func):
        """Applies an operation that takes only the current processed image."""
        if self._ensure_image_loaded():
            self.apply_operation(operation_func)

    def apply_operation_with_params(self, operation_func, params_desc):
        """Gets parameters from user via QInputDialog and applies the operation."""
        if not self._ensure_image_loaded():
            return

        args = []
        for desc in params_desc:
            # label, type_str, min_val, max_val, default_val, *rest = desc (original)
            # Unpack carefully, step/decimals might not always be present in `rest`
            label, type_str, min_val, max_val, default_val = desc[:5]
            rest = desc[5:]
            
            val, ok = None, False
            if type_str == "int":
                 step = rest[0] if rest else 1
                 val, ok = QInputDialog.getInt(self, "Input Parameter", label, default_val, min_val, max_val, step)
            elif type_str == "double":
                 decimals = rest[0] if rest else 2 # Default to 2 decimals if not specified
                 val, ok = QInputDialog.getDouble(self, "Input Parameter", label, default_val, min_val, max_val, decimals)
            else:
                QMessageBox.critical(self, "Error", f"Unsupported parameter type '{type_str}' in function definition.")
                return # Abort operation
            
            if ok:
                args.append(val)
            else: # User cancelled an input dialog
                return 
                
        if args or not params_desc: # Proceed if args collected or if no params were needed (though this path is for param_ops)
             self.apply_operation(operation_func, *args)

    def apply_operation(self, operation_func, *args):
        """The core worker method that applies an image operation and updates the display."""
        if not self._ensure_image_loaded(): # Should be redundant if callers check, but good for safety
            return

        try:
            print(f"Applying {operation_func.__name__} with args: {args}") # Debug
            if self.processed_image is not None:
                result = operation_func(self.processed_image.copy(), *args)
            else:
                QMessageBox.critical(self, "Operation Error", "No image loaded to apply the operation.")
                return

            if isinstance(result, np.ndarray) and result.ndim in [2, 3]:
                 self.processed_image = result
                 self.display_image(self.processed_image)
                 self.dft_shift = None # Image changed, invalidate DFT cache
                 self.spectrum_button.setEnabled(False)
            elif result is not None: # Operation returned something, but not a displayable image
                 QMessageBox.information(self, "Operation Note", 
                                         f"Operation {operation_func.__name__} executed, but did not return a new displayable image. Result type: {type(result)}")
            # If result is None, it implies the operation might have failed silently or had no return; error caught below
        except Exception as e:
            error_message = f"Error applying {operation_func.__name__}:\n{e}"
            print(error_message) # Debug
            QMessageBox.critical(self, "Operation Error", error_message)

    # --- Specific Feature Handlers ---
    def run_calculate_dft(self):
        if self._ensure_image_loaded():
            try:
                if self.processed_image is not None:
                    self.dft_shift = img_ops.calculate_dft(self.processed_image.copy())
                    self.spectrum_button.setEnabled(True)
                    QMessageBox.information(self,"DFT Calculation", "DFT calculated successfully. You can now show the magnitude spectrum.")
                else:
                    QMessageBox.critical(self, "DFT Error", "No image loaded to calculate DFT.")
                    self.dft_shift = None
                    self.spectrum_button.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "DFT Error", f"Failed to calculate DFT:\n{e}")
                self.dft_shift = None
                self.spectrum_button.setEnabled(False)

    def show_magnitude_spectrum(self):
        if self.dft_shift is None:
             QMessageBox.warning(self, "Magnitude Spectrum", "Please calculate DFT first using the 'Calculate DFT' button.")
             return
        if self._ensure_image_loaded(): # Ensure there's a base image context, though dft_shift is key
            try:
                 spectrum = img_ops.calculate_magnitude_spectrum(self.dft_shift)
                 if spectrum is not None:
                     self.processed_image = spectrum # Display spectrum as current image
                     self.display_image(self.processed_image)
                 else:
                     QMessageBox.warning(self, "Spectrum Error", "Failed to generate magnitude spectrum image.")
            except Exception as e:
                 QMessageBox.critical(self, "Spectrum Error", f"Failed to calculate/display magnitude spectrum:\n{e}")

    def show_histogram(self):
         if self._ensure_image_loaded():
             try:
                 hist_result = img_ops.calculate_histogram(self.processed_image)
                 plt.figure(figsize=(8, 5))
                 if isinstance(hist_result, tuple) and len(hist_result) == 3:
                     hist_r, hist_g, hist_b = hist_result
                     plt.plot(hist_r, color='r', label='Red')
                     plt.plot(hist_g, color='g', label='Green')
                     plt.plot(hist_b, color='b', label='Blue')
                     plt.title('RGB Channel Histograms')
                     plt.xlabel('Pixel Value')
                     plt.ylabel('Frequency')
                     plt.legend()
                 else:
                     plt.plot(hist_result, color='k', label='Grayscale')
                     plt.title('Grayscale Histogram')
                     plt.xlabel('Pixel Value')
                     plt.ylabel('Frequency')
                     plt.legend()
                 plt.tight_layout()
                 plt.show()
             except Exception as e:
                 QMessageBox.critical(self, "Histogram Error", f"Failed to calculate/display histogram:\n{e}")

    def show_split_channels(self):
        if self._ensure_image_loaded():
            if self.processed_image is None or self.processed_image.ndim != 3:
                QMessageBox.warning(self, "Split Channels", "This operation requires a color image.")
                return
            try:
                r, g, b = img_ops.split_rgb_channels(self.processed_image)
                # Show each channel using matplotlib
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(r, cmap='Reds')
                axs[0].set_title('Red Channel')
                axs[1].imshow(g, cmap='Greens')
                axs[1].set_title('Green Channel')
                axs[2].imshow(b, cmap='Blues')
                axs[2].set_title('Blue Channel')
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                QMessageBox.critical(self, "Split Channels Error", f"Failed to split/display channels:\n{e}")

    # --- Utility and Reset Methods ---
    def reset_image(self):
        if self.current_image is not None:
            self.processed_image = self.current_image.copy()
            self.display_image(self.processed_image)
            self.dft_shift = None 
            self.spectrum_button.setEnabled(False)
            QMessageBox.information(self, "Image Reset", "Image has been reset to the last loaded version.")
        else:
             QMessageBox.warning(self, "Reset Image", "No image loaded to reset.")

    def _set_operation_buttons_enabled(self, enabled: bool):
        """Enables or disables all operation-related buttons."""
        # Iterate through QGridLayout items
        for i in range(self.functions_layout.count()):
            item = self.functions_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if isinstance(widget, QPushButton):
                    # Don't disable load/save here, they have their own logic
                    # The "Show Mag Spectrum" button has special enable logic
                    if widget not in [self.load_button, self.save_button, self.spectrum_button]:
                        widget.setEnabled(enabled)
        # Reset button specifically:
        reset_button_item = self.functions_layout.itemAt(self.functions_layout.count() - 1)
        if reset_button_item is not None:
            reset_button_widget = reset_button_item.widget()
            if isinstance(reset_button_widget, QPushButton) and reset_button_widget.text() == "Reset to Original":
                reset_button_widget.setEnabled(enabled if self.current_image is not None else False)


    def _ensure_image_loaded(self) -> bool:
        """Checks if an image is loaded and shows a warning if not."""
        if self.processed_image is None:
            QMessageBox.warning(self, "Image Required", "Please load an image first to perform this operation.")
            return False
        return True

if __name__ == "__main__":
    app = QApplication.instance() # Check if an instance already exists
    if not app:
        app = QApplication(sys.argv)
    
    # Apply High DPI scaling for better visuals on supported displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough) 
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())