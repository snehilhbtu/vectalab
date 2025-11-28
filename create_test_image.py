import cv2
import numpy as np

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Draw a red rectangle
cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), -1)

# Draw a green circle
cv2.circle(img, (300, 300), 100, (0, 255, 0), -1)

# Draw a blue triangle
pts = np.array([[400, 50], [300, 200], [500, 200]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(img, [pts], (255, 0, 0))

# Save
cv2.imwrite("test_image.png", img)
print("Created test_image.png")
