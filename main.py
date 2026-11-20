
import cv2
import numpy as np
import matplotlib

# Use a non-GUI backend to avoid the Tkinter error
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os


# ==========================================
# PROJECT 1: DIGITAL IMAGE MANIPULATION
# AND ENHANCEMENT SYSTEM
# ==========================================

IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "outputs"

IMAGE_NAMES = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def save_image(path, image):
    """Save an image and report any errors."""

    success = cv2.imwrite(path, image)

    if not success:
        print("Error: Could not save:", path)

    return success


def save_plot(path):
    """Save the current Matplotlib figure."""

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def display_rgb(image):
    """Convert BGR image to RGB for Matplotlib."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================
# STEP 1: READ AND DISPLAY IMAGES
# ==========================================

def read_images():

    images = []

    print("\n========== IMAGE READING ==========")

    for name in IMAGE_NAMES:

        path = os.path.join(IMAGE_FOLDER, name)

        image = cv2.imread(path)

        if image is None:
            print("Error: Cannot read", name)
            continue

        images.append((name, image))

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        print("\nImage:", name)
        print("Width:", width)
        print("Height:", height)
        print("Channels:", channels)

        # Display center pixel
        y = height // 2
        x = width // 2

        print("Center Pixel (BGR):", image[y, x])

        # Create output folder for this image
        image_output = os.path.join(
            OUTPUT_FOLDER,
            os.path.splitext(name)[0]
        )

        os.makedirs(image_output, exist_ok=True)

        # Display and save original image
        plt.figure(figsize=(7, 5))
        plt.imshow(display_rgb(image))
        plt.title("Original Image: " + name)
        plt.axis("off")

        save_plot(
            os.path.join(image_output, "original.png")
        )

        # Horizontal flip
        horizontal = cv2.flip(image, 1)

        save_image(
            os.path.join(image_output, "horizontal_flip.jpg"),
            horizontal
        )

        # Vertical flip
        vertical = cv2.flip(image, 0)

        save_image(
            os.path.join(image_output, "vertical_flip.jpg"),
            vertical
        )

        # Save flip comparison
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(display_rgb(horizontal))
        plt.title("Horizontal Flip")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(display_rgb(vertical))
        plt.title("Vertical Flip")
        plt.axis("off")

        save_plot(
            os.path.join(image_output, "flip_comparison.png")
        )

    print("\nTotal Images Read:", len(images))

    return images


# ==========================================
# STEP 2: GRAYSCALE AND GRAY LEVEL REDUCTION
# ==========================================

def gray_level_reduction(image, output_folder):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    levels = [256, 128, 64, 32, 16, 8, 4, 2]

    plt.figure(figsize=(15, 10))

    for i, level in enumerate(levels):

        if level == 256:
            reduced = gray.copy()

        else:
            # Quantize the grayscale image to the required levels
            reduced = np.round(
                gray.astype(np.float32) * (level - 1) / 255
            ) * (255 / (level - 1))

            reduced = np.uint8(
                np.clip(reduced, 0, 255)
            )

        # Display result
        plt.subplot(2, 4, i + 1)

        plt.imshow(
            reduced,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        plt.title(f"{level} Gray Levels")
        plt.axis("off")

        # Save result
        save_image(
            os.path.join(
                output_folder,
                f"gray_{level}_levels.jpg"
            ),
            reduced
        )

    save_plot(
        os.path.join(
            output_folder,
            "gray_level_comparison.png"
        )
    )


# ==========================================
# STEP 3: IMAGE RESIZING
# ==========================================

def image_resizing(image, output_folder):

    height, width = image.shape[:2]

    # Check image dimensions
    if width < 2 or height < 2:
        print("Image is too small for resizing.")
        return

    # Enlargement dimensions
    large_size = (width * 2, height * 2)

    # Reduction dimensions
    small_size = (
        max(1, width // 2),
        max(1, height // 2)
    )

    # Nearest Neighbor Enlargement
    nearest = cv2.resize(
        image,
        large_size,
        interpolation=cv2.INTER_NEAREST
    )

    # Bilinear Enlargement
    bilinear = cv2.resize(
        image,
        large_size,
        interpolation=cv2.INTER_LINEAR
    )

    # Nearest Neighbor Reduction
    nearest_small = cv2.resize(
        image,
        small_size,
        interpolation=cv2.INTER_NEAREST
    )

    # Bilinear Reduction
    bilinear_small = cv2.resize(
        image,
        small_size,
        interpolation=cv2.INTER_LINEAR
    )

    # Save resized images
    save_image(
        os.path.join(output_folder, "nearest_large.jpg"),
        nearest
    )

    save_image(
        os.path.join(output_folder, "bilinear_large.jpg"),
        bilinear
    )

    save_image(
        os.path.join(output_folder, "nearest_small.jpg"),
        nearest_small
    )

    save_image(
        os.path.join(output_folder, "bilinear_small.jpg"),
        bilinear_small
    )

    # Display comparison
    images = [
        ("Nearest Enlargement", nearest),
        ("Bilinear Enlargement", bilinear),
        ("Nearest Reduction", nearest_small),
        ("Bilinear Reduction", bilinear_small)
    ]

    plt.figure(figsize=(12, 9))

    for i, (title, img) in enumerate(images):

        plt.subplot(2, 2, i + 1)

        plt.imshow(display_rgb(img))
        plt.title(title)
        plt.axis("off")

    save_plot(
        os.path.join(
            output_folder,
            "resizing_comparison.png"
        )
    )


# ==========================================
# STEP 4: INTENSITY TRANSFORMATIONS
# ==========================================

def intensity_transformations(image, output_folder):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------
    # 1. IMAGE NEGATIVE
    # --------------------------------------

    negative = 255 - gray

    save_image(
        os.path.join(output_folder, "negative.jpg"),
        negative
    )

    # --------------------------------------
    # 2. LOGARITHMIC TRANSFORMATION
    # --------------------------------------

    max_value = np.max(gray)

    if max_value > 0:

        c = 255 / np.log1p(max_value)

        logarithmic = c * np.log1p(
            gray.astype(np.float32)
        )

    else:
        logarithmic = gray.astype(np.float32)

    logarithmic = np.uint8(
        np.clip(logarithmic, 0, 255)
    )

    save_image(
        os.path.join(output_folder, "logarithmic.jpg"),
        logarithmic
    )

    # --------------------------------------
    # 3. GAMMA TRANSFORMATION
    # --------------------------------------

    gamma_values = [0.5, 1.0, 1.5, 2.0]

    plt.figure(figsize=(12, 9))

    for i, gamma in enumerate(gamma_values):

        normalized = gray.astype(
            np.float32
        ) / 255.0

        gamma_image = (
            np.power(normalized, gamma) * 255
        )

        gamma_image = np.uint8(
            np.clip(gamma_image, 0, 255)
        )

        save_image(
            os.path.join(
                output_folder,
                f"gamma_{gamma}.jpg"
            ),
            gamma_image
        )

        plt.subplot(2, 2, i + 1)

        plt.imshow(
            gamma_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        plt.title(f"Gamma = {gamma}")
        plt.axis("off")

    save_plot(
        os.path.join(
            output_folder,
            "gamma_comparison.png"
        )
    )

    # --------------------------------------
    # 4. NEGATIVE AND LOG COMPARISON
    # --------------------------------------

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(negative, cmap="gray")
    plt.title("Image Negative")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(logarithmic, cmap="gray")
    plt.title("Logarithmic Transformation")
    plt.axis("off")

    save_plot(
        os.path.join(
            output_folder,
            "intensity_comparison.png"
        )
    )


# ==========================================
# STEP 5: HISTOGRAM PROCESSING
# ==========================================

def histogram_processing(image, output_folder):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------
    # 1. ORIGINAL HISTOGRAM
    # --------------------------------------

    original_hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    # --------------------------------------
    # 2. HISTOGRAM EQUALIZATION
    # --------------------------------------

    equalized = cv2.equalizeHist(gray)

    # --------------------------------------
    # 3. EQUALIZED HISTOGRAM
    # --------------------------------------

    equalized_hist = cv2.calcHist(
        [equalized],
        [0],
        None,
        [256],
        [0, 256]
    )

    # Save images
    save_image(
        os.path.join(output_folder, "original_gray.jpg"),
        gray
    )

    save_image(
        os.path.join(output_folder, "equalized.jpg"),
        equalized
    )

    # --------------------------------------
    # 4. DISPLAY COMPARISON
    # --------------------------------------

    plt.figure(figsize=(12, 9))

    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(equalized, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.plot(original_hist.flatten())
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])

    plt.subplot(2, 2, 4)
    plt.plot(equalized_hist.flatten())
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])

    save_plot(
        os.path.join(
            output_folder,
            "histogram_comparison.png"
        )
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("===================================")
    print("DIGITAL IMAGE MANIPULATION SYSTEM")
    print("===================================")

    # Read all images
    images = read_images()

    if len(images) == 0:

        print("\nNo images found!")
        print("Please check the images folder.")
        return

    # Process each available image
    for name, image in images:

        image_output = os.path.join(
            OUTPUT_FOLDER,
            os.path.splitext(name)[0]
        )

        print("\n===================================")
        print("PROCESSING:", name)
        print("===================================")

        print("\nRunning Grayscale Processing...")
        gray_level_reduction(
            image,
            image_output
        )

        print("Grayscale Processing Completed.")

        print("\nRunning Image Resizing...")
        image_resizing(
            image,
            image_output
        )

        print("Image Resizing Completed.")

        print("\nRunning Intensity Transformations...")
        intensity_transformations(
            image,
            image_output
        )

        print("Intensity Transformations Completed.")

        print("\nRunning Histogram Processing...")
        histogram_processing(
            image,
            image_output
        )

        print("Histogram Processing Completed.")

    print("\n===================================")
    print("ALL OPERATIONS COMPLETED!")
    print("===================================")

    print("Check the outputs folder for results.")


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
