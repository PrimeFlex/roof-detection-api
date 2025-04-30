import cv2
import numpy as np

# --- CONFIG ---
INPUT_PATH = "roof_input.png"  # Replace with your test image
OUTPUT_PATH = "roof_sketch_output.png"

# --- Load image ---
image = cv2.imread(INPUT_PATH)
assert image is not None, f"Failed to load image: {INPUT_PATH}"
print("✅ Image loaded.")

# --- Resize for consistency ---
scale_percent = 70
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized = cv2.resize(image, (width, height))

# --- Convert to grayscale and blur ---
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Edge detection ---
edged = cv2.Canny(blurred, 30, 100)

# --- Morph ops to close gaps ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(edged, kernel, iterations=1)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# --- Find contours ---
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"🧠 {len(contours)} contours found")

# --- Filter and draw large contours ---
MIN_AREA = 1000
sketch = resized.copy()
count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > MIN_AREA:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(sketch, [approx], -1, (0, 255, 0), 2)
        count += 1

print(f"✅ {count} clean contours drawn")
cv2.imwrite(OUTPUT_PATH, sketch)
print(f"📝 Saved to: {OUTPUT_PATH}")
