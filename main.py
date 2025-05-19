from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image_BGR(image_path): # Read image in BGR mode (default for OpenCV)
    image = cv2.imread(str(image_path))
    return image

def read_image_gray(image_path): # Read image in grayscale mode
    image = cv2.imread(str(image_path), 0)
    return image

def convert_to_gray(image): # Convert BGR image to grayscale
    if len(image.shape) == 3:  # Check if the image is in color (3 channels)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image2
    else:
        return image

def save_image(image, output_path):
    cv2.imwrite(str(output_path), image)
    return True

def get_negative(image):
    negative = cv2.bitwise_not(image)
    return negative

def split_rgb_channels(image):
    if len(image.shape) != 3:
        raise ValueError("Image must be a color image with 3 channels")
    b, g, r = cv2.split(image)
    return r, g, b

def adjust_brightness(image, value):
    """Adjust brightness of an image."""
    if value == 0:
        return image
    
    adjusted = np.clip(image.astype(np.int16) + value, 0, 255).astype(np.uint8)
    return adjusted

def apply_threshold(image, threshold_value, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """Apply thresholding to an image."""
    
    if len(image.shape) == 3:
        image = convert_to_gray(image)
    
    _, thresholded = cv2.threshold(image, threshold_value, max_value, threshold_type)
    return thresholded

def calculate_histogram(image):
    """Calculate histogram of an image."""
    if len(image.shape) == 3:
        # For color images, calculate histogram for each channel
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist_r, hist_g, hist_b
    else:
        # For grayscale images
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist

def equalize_histogram(image):
    """Apply histogram equalization to enhance contrast."""
    if len(image.shape) == 3:
        # For color images, convert to YCrCb, equalize Y channel, then convert back
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return equalized
    else:
        # For grayscale images
        return cv2.equalizeHist(image)

def adjust_contrast(image, alpha, beta=0):
    """Adjust contrast of an image using the formula: new_img = alpha*img + beta
    
    Args:
        image: Input image
        alpha: Contrast control (1.0-3.0: increase contrast, 0.0-1.0: decrease contrast)
        beta: Brightness control (0-100)
    
    Returns:
        Contrast adjusted image
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def translate_image(image, x, y):
    """Translate an image by x and y pixels."""
    rows, cols = image.shape[:2]
    M = np.array([[1, 0, x], [0, 1, y]], dtype=np.float32)
    translated = cv2.warpAffine(image, M, (cols, rows))
    return translated

def flip_image(image, flip_code):
    """Flip an image.
    
    Args:
        image: Input image
        flip_code: 0 for vertical flip, 1 for horizontal flip, -1 for both
    
    Returns:
        Flipped image
    """
    return cv2.flip(image, flip_code)

def shear_image(image, shear_factor_x=0, shear_factor_y=0):
    """Apply shearing transformation to an image."""
    rows, cols = image.shape[:2]
    
    # Create shear matrix
    M = np.array([
        [1, shear_factor_x, 0],
        [shear_factor_y, 1, 0]
    ], dtype=np.float32)
    
    # Apply affine transformation
    sheared = cv2.warpAffine(image, M, (cols, rows))
    return sheared

def resize_image(image, scale_factor=None, width=None, height=None):
    """Resize an image by scale factor or to specific dimensions."""
    if scale_factor is not None:
        resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    elif width is not None and height is not None:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError("Either scale_factor or both width and height must be provided")
    return resized

def rotate_image(image, angle, center=None, scale=1.0):
    """Rotate an image by a given angle."""
    rows, cols = image.shape[:2]
    if center is None:
        center = (cols // 2, rows // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    return rotated

def crop_image(image, x, y, width, height):
    """Crop a region from an image."""
    return image[y:y+height, x:x+width]

def apply_mean_filter(image, kernel_size=3):
    """Apply mean (average) filter to an image."""
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_median_filter(image, kernel_size=3):
    """Apply median filter to an image."""
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size=3, sigma=0):
    """Apply Gaussian filter to an image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_conservative_filter(image):
    """Apply conservative filter to an image."""
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):  # Process each channel
            channel = image[:,:,i]
            filtered_channel = np.zeros_like(channel)
            rows, cols = channel.shape
            for r in range(1, rows-1):
                for c in range(1, cols-1):
                    neighborhood = channel[r-1:r+2, c-1:c+2].flatten()
                    center = channel[r, c]
                    max_val = np.max(neighborhood)
                    min_val = np.min(neighborhood)
                    if center > max_val:
                        filtered_channel[r, c] = max_val
                    elif center < min_val:
                        filtered_channel[r, c] = min_val
                    else:
                        filtered_channel[r, c] = center
            result[:,:,i] = filtered_channel
    else:
        result = np.zeros_like(image)
        rows, cols = image.shape
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                neighborhood = image[r-1:r+2, c-1:c+2].flatten()
                center = image[r, c]
                max_val = np.max(neighborhood)
                min_val = np.min(neighborhood)
                if center > max_val:
                    result[r, c] = max_val
                elif center < min_val:
                    result[r, c] = min_val
                else:
                    result[r, c] = center
    return result

def apply_sobel_edge_detection(image, dx=1, dy=1, ksize=3):
    """Apply Sobel edge detection to an image.
    If dx=1 and dy=1, computes the magnitude of edges.
    If dx=1, dy=0, computes x-derivative.
    If dx=0, dy=1, computes y-derivative.
    """
    if len(image.shape) == 3:
        image = convert_to_gray(image)

    img_uint8 = image.astype(np.uint8)

    grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, dx, 0, ksize=ksize)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, dy, ksize=ksize)

    if dx == 1 and dy == 1: # Typical case for edge magnitude
        magnitude = cv2.magnitude(grad_x, grad_y)
        return cv2.convertScaleAbs(magnitude)
    elif dx == 1: # Only X derivative
        return cv2.convertScaleAbs(grad_x)
    elif dy == 1: # Only Y derivative
        return cv2.convertScaleAbs(grad_y)
    else:
        # Should not happen if GUI enforces dx or dy is 1
        return image.astype(np.uint8) # Or a black image

def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
    """Apply Canny edge detection to an image."""
    if len(image.shape) == 3:
        image = convert_to_gray(image)
    return cv2.Canny(image, threshold1, threshold2)

def apply_erosion(image, kernel_size=3, iterations=1):
    """Apply erosion morphological operation to an image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def apply_dilation(image, kernel_size=3, iterations=1):
    """Apply dilation morphological operation to an image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

# --- Added Edge Detectors ---

def apply_prewitt_edge_detection(image):
    """Apply Prewitt edge detection."""
    if len(image.shape) == 3:
        image = convert_to_gray(image)
    
    img_float = image.astype(np.float32)
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    
    # Change CV_64F to CV_32F for the output depth of filter2D
    grad_x = cv2.filter2D(img_float, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(img_float, cv2.CV_32F, kernel_y)
    
    # Combine gradients - magnitude
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize the magnitude to the 0-255 range to enhance visibility
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    
    edge_image = cv2.convertScaleAbs(magnitude)
    
    return edge_image

def apply_roberts_cross_edge_detection(image):
    """Apply Roberts Cross edge detection."""
    if len(image.shape) == 3:
        image = convert_to_gray(image)

    img_float = image.astype(np.float32)
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Change CV_64F to CV_32F for the output depth of filter2D
    grad_x = cv2.filter2D(img_float, cv2.CV_32F, kernel_x, anchor=(0, 0))
    grad_y = cv2.filter2D(img_float, cv2.CV_32F, kernel_y, anchor=(0, 0))

    magnitude = cv2.magnitude(grad_x, grad_y)
    # Normalize before converting to uint8 for better visualization
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    magnitude = cv2.convertScaleAbs(magnitude)
    return magnitude

def apply_laplacian_edge_detection(image, ksize=3):
    """Apply Laplacian edge detection."""
    if len(image.shape) == 3:
        image = convert_to_gray(image)
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    return laplacian_abs

# --- Added Segmentation ---

def apply_kmeans_segmentation(image, k=3):
    """Apply K-Means segmentation."""
    if image is None: return None 

    original_shape = image.shape
    if len(original_shape) != 3:
        print("K-means segmentation is typically applied to color images. Converting grayscale to BGR.")
        image_for_kmeans = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_for_kmeans = image

    pixel_values = image_for_kmeans.reshape((-1, 3)) 
    pixel_values = pixel_values.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # Create a dummy labels array for initial labels as required by OpenCV's kmeans
    if pixel_values.size == 0:
        raise ValueError("Pixel values are empty. Ensure the input image is valid and not empty.")
    bestLabels = np.zeros((len(pixel_values), 1), dtype=np.int32)
    _, labels, centers = cv2.kmeans(np.array(pixel_values, dtype=np.float32), k, bestLabels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = centers.astype(np.uint8) 
    segmented_data = centers[labels.flatten().astype(np.int32)] 
    segmented_image = segmented_data.reshape(image_for_kmeans.shape) 

    if len(original_shape) == 2 and len(segmented_image.shape) == 3:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        
    return segmented_image

# --- Added Frequency Domain Basics ---

def calculate_dft(image):
    """Calculate the Discrete Fourier Transform of an image."""
    if len(image.shape) == 3:
        image_gray = convert_to_gray(image)
    else:
        image_gray = image
    
    # Convert to UMat for DFT, then back to numpy for fftshift
    umat = cv2.UMat(image_gray.astype(np.float32))
    dft_result = cv2.dft(umat, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_np = dft_result.get()
    dft_shift = np.fft.fftshift(dft_np)
    return dft_shift # Return the shifted spectrum

def calculate_magnitude_spectrum(dft_shift):
    """Calculate the magnitude spectrum from a shifted DFT."""
    if dft_shift is None: return None
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = 20 * np.log(magnitude + 1) # +1 to avoid log(0)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, magnitude_spectrum, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return magnitude_spectrum

# --- Added placeholder for interactive perspective ---
# This needs GUI interaction, so we just define the core transform function here
def apply_perspective_transform(image, src_points, dst_points, output_size):
    """Applies perspective transform given source and destination points."""
    if not isinstance(src_points, np.ndarray) or src_points.shape != (4, 2):
        raise ValueError("src_points must be a 4x2 numpy array of floats")
    if not isinstance(dst_points, np.ndarray) or dst_points.shape != (4, 2):
        raise ValueError("dst_points must be a 4x2 numpy array of floats")
        
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    warped = cv2.warpPerspective(image, M, output_size) # output_size is (width, height)
    return warped

# Note: Functions like Fourier LPF/HPF, Butterworth, Homomorphic, Gabor, Hough, etc., 
# require more involved implementation or specific parameters and are deferred for now.
