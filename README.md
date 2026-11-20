# Digital Image Manipulation and Enhancement System

This project demonstrates common digital image processing operations using Python,
OpenCV, NumPy, and Matplotlib. It reads three sample images, applies several
transformations, and saves both the processed images and visual comparison plots
under the `outputs` directory.

## Features

- Reads and inspects color images
- Prints image width, height, channel count, and center pixel value
- Creates horizontal and vertical flips
- Converts images to grayscale with 256, 128, 64, 32, 16, 8, 4, and 2 gray levels
- Resizes images using nearest-neighbor and bilinear interpolation
- Applies image negative and logarithmic intensity transformations
- Applies gamma transformations with gamma values of 0.5, 1.0, 1.5, and 2.0
- Generates original and equalized grayscale histograms
- Applies histogram equalization
- Saves individual results and side-by-side comparison plots

## Technologies

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Project Structure

```text
Digital_Image_Manipulation/
|-- images/
|   |-- image1.jpg
|   |-- image2.jpg
|   `-- image3.jpg
|-- outputs/
|   |-- image1/
|   |-- image2/
|   `-- image3/
|-- main.py
|-- requirements.txt
`-- README.md
```

## Installation

1. Open a terminal in the project directory.

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install the required packages:

```powershell
pip install -r requirements.txt
```

## Input Images

Place the images to process in the `images` folder. The program currently expects:

- `images/image1.jpg`
- `images/image2.jpg`
- `images/image3.jpg`

To use different filenames, update the `IMAGE_NAMES` list in `main.py`.

## Usage

Run the program from the project directory:

```powershell
python main.py
```

The program uses a non-GUI Matplotlib backend, so all visual results are saved
as image files rather than opened in separate windows.

## Output

Processed files are written to `outputs/<image-name>/`. For example,
`outputs/image1/` contains:

| Output | Description |
|---|---|
| `original.png` | Original image |
| `horizontal_flip.jpg` | Horizontally flipped image |
| `vertical_flip.jpg` | Vertically flipped image |
| `gray_*_levels.jpg` | Images quantized to a specific number of gray levels |
| `nearest_large.jpg` | Nearest-neighbor enlargement |
| `bilinear_large.jpg` | Bilinear enlargement |
| `nearest_small.jpg` | Nearest-neighbor reduction |
| `bilinear_small.jpg` | Bilinear reduction |
| `negative.jpg` | Negative grayscale image |
| `logarithmic.jpg` | Logarithmic intensity transformation |
| `gamma_*.jpg` | Gamma-corrected grayscale images |
| `equalized.jpg` | Histogram-equalized grayscale image |
| `*_comparison.png` | Side-by-side visual comparisons |

## Processing Summary

1. **Image reading:** The images are loaded with OpenCV and their basic properties are printed.
2. **Flipping:** Images are reversed horizontally and vertically.
3. **Gray-level reduction:** Grayscale levels are quantized to fewer intensity levels.
4. **Resizing:** Images are enlarged and reduced with two interpolation methods.
5. **Intensity transformations:** Negative, logarithmic, and gamma transformations are applied.
6. **Histogram processing:** Histograms are calculated before and after histogram equalization.

## Troubleshooting

- If the program reports that an image cannot be read, verify the filename and its location in `images`.
- If an import error occurs, activate the virtual environment and install the dependencies again.
- If output files are not created, confirm that Python can write to the project directory.