import cv2
import numpy as np

def process_and_save_image(image, final_width=800, final_height=1000):
    """
    Processes the given image by detecting, cropping, and resizing the exam sheet.
    
    Parameters:
        image (np.array): Image array uploaded by user.
        final_width (int): Desired width of the output image.
        final_height (int): Desired height of the output image.
    
    Returns:
        np.array: Processed and resized image.
    """
    processed_image = process_image(image, final_width, final_height)
    return processed_image

def process_image(image, final_width, final_height):
    """
    Converts the image to grayscale, detects edges, crops the largest contour (exam sheet),
    and resizes it to the specified dimensions.

    Parameters:
        image (np.array): Input image array.
        final_width (int): Width to resize the output image.
        final_height (int): Height to resize the output image.
    
    Returns:
        np.array: Processed and resized image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        # Largest contour (assumed to be the exam sheet)
        sheet_contour = contours[0]
        epsilon = 0.02 * cv2.arcLength(sheet_contour, True)
        approx = cv2.approxPolyDP(sheet_contour, epsilon, True)

        if len(approx) == 4:  # Ensure it's a quadrilateral
            # Order the points correctly
            rect = np.float32(order_points(approx.reshape(4, 2)))
            
            # Compute the width and height of the new image
            width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
            height = max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0]))
            
            # Define the destination points for warping
            dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, matrix, (int(width), int(height)))

            # Resize the image to a fixed size (800x1000 pixels by default)
            resized = cv2.resize(warped, (final_width, final_height), interpolation=cv2.INTER_AREA)

            return resized

    print("No suitable contour found. Returning original image.")
    return cv2.resize(image, (final_width, final_height), interpolation=cv2.INTER_AREA)

def order_points(pts):
    """
    Orders points in the following sequence: top-left, top-right, bottom-right, bottom-left.

    Parameters:
        pts (np.array): An array containing four coordinate points.

    Returns:
        np.array: Ordered array of points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect
