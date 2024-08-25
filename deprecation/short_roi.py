import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def draw_polygon(image, points, color, thickness):
    points = points.astype(np.int32)  # Convert points to int32
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

def put_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness)

image = cv2.imread('3.png')

# 1左上角:195,285  =>522,286  =>535,656  =>209,669
# 2左下角:574,393 8=>1657,3702=>1990,5412=>809,5769
# 3右下角:2520,3388=>6059,2762=>6746,4364=>3111,5636
points1 = np.array([[195, 285], [522, 286], [535, 656], [209, 669]], dtype=np.float32)
points2 = np.array([[574, 3938], [1657, 3702], [1990, 5412], [809, 5769]], dtype=np.float32)
points3 = np.array([[2520, 3388], [6059, 2762], [6746, 4364], [3111, 5636]], dtype=np.float32)
# points1 = np.array([[139, 188], [986, 250], [683, 522], [164, 522]], dtype=np.float32)
# points2 = np.array([[164, 522], [683, 522], [423, 884], [213, 884]], dtype=np.float32)
# points3 = np.array([[213, 884], [423, 884], [321, 1214], [250, 1214]], dtype=np.float32)

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

# Draw the polygons on the original image
draw_polygon(image, points1, color=(0, 255, 0), thickness=3)  # Green for area1
draw_polygon(image, points2, color=(255, 0, 0), thickness=3)  # Blue for area2
draw_polygon(image, points3, color=(0, 0, 255), thickness=3)  # Red for area3

# Add text labels to the original image
put_text(image, 'Area 1', (139, 188 - 10), color=(0, 255, 0))
put_text(image, 'Area 2', (164, 522 - 10), color=(255, 0, 0))
put_text(image, 'Area 3', (213, 884 - 10), color=(0, 0, 255))

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

# Display images with OpenCV
cv2.imshow('Input', cv2.resize(image, None, fx=0.1, fy=0.1))
cv2.imshow('Area1', dst1)
cv2.imshow('Area2', dst2)
cv2.imshow('Area3', dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the images with drawn areas
# cv2.imwrite('image_with_areas.png', image)
# cv2.imwrite('transformed_area1.png', dst1)
# cv2.imwrite('transformed_area2.png', dst2)
# cv2.imwrite('transformed_area3.png', dst3)
