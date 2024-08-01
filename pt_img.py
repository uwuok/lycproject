import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
直接硬 perspective transformation，會導致圖片在經過拉伸後的變形 
'''

def get_bounding_box_corners(points):
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return np.array([
        [0, 0], [400, 0], [400, 300], [0, 300]
        # [np.min(x_coords), np.min(y_coords)],
        # [np.max(x_coords), np.min(y_coords)],
        # [np.max(x_coords), np.max(y_coords)],
        # [np.min(x_coords), np.max(y_coords)]
    ], dtype=np.float32)

def calculate_output_size(points, transform_matrix):
    transformed_points = cv2.perspectiveTransform(np.array([points]), transform_matrix)[0]
    bounding_box = get_bounding_box_corners(transformed_points)
    width = np.max(bounding_box[:, 0]) - np.min(bounding_box[:, 0])
    height = np.max(bounding_box[:, 1]) - np.min(bounding_box[:, 1])
    return int(width), int(height), transformed_points

image = cv2.imread('3.png')

points1 = np.array([[139, 188], [986, 250], [683, 522], [164, 522]], dtype=np.float32)
points2 = np.array([[164, 522], [683, 522], [423, 884], [213, 884]], dtype=np.float32)
points3 = np.array([[213, 884], [423, 884], [321, 1214], [250, 1214]], dtype=np.float32)

# Calculate the new_points for bounding box
new_points1 = get_bounding_box_corners(points1)
new_points2 = get_bounding_box_corners(points2)
new_points3 = get_bounding_box_corners(points3)

# Perspective transformation for points1
m1 = cv2.getPerspectiveTransform(points1, new_points1)
width1, height1, transformed_points1 = calculate_output_size(points1, m1)
dst1 = cv2.warpPerspective(image, m1, (width1, height1))

# Perspective transformation for points2
m2 = cv2.getPerspectiveTransform(points2, new_points2)
width2, height2, transformed_points2 = calculate_output_size(points2, m2)
dst2 = cv2.warpPerspective(image, m2, (width2, height2))

# Perspective transformation for points3
m3 = cv2.getPerspectiveTransform(points3, new_points3)
width3, height3, transformed_points3 = calculate_output_size(points3, m3)
dst3 = cv2.warpPerspective(image, m3, (width3, height3))

# Plot original and transformed images with points
def plot_image_with_points(img, points, title):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], color='red')
    plt.title(title)
    plt.show()

# Original image
# plot_image_with_points(image, points1, 'Original Image with Points1')
# plot_image_with_points(image, points2, 'Original Image with Points2')
# plot_image_with_points(image, points3, 'Original Image with Points3')

# Transformed images
# plot_image_with_points(dst1, transformed_points1, 'Transformed Image 1 with Points')
# plot_image_with_points(dst2, transformed_points2, 'Transformed Image 2 with Points')
# plot_image_with_points(dst3, transformed_points3, 'Transformed Image 3 with Points')

# Display images with fx=1 and fy=1
cv2.imshow('Input', cv2.resize(image, None, fx=0.3, fy=0.3))
cv2.imshow('Area1', dst1)
cv2.imshow('Area2', dst2)
cv2.imshow('Area3', dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()
