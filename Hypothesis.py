import numpy as np
from PIL import Image


# Load the original RGB image
common_path = "/content/"
#original_image = Image.open(common_path + "Ishihara_12.jpg")
original_image = Image.open("/content/besttest.jpeg")
original_array = np.array(original_image)

# RGB to LMS transformation matrix (equation M4)
rgb_to_lms_matrix = np.array([[17.8824, 43.5161, 4.11935],
                              [3.45565, 27.1554, 3.86714],
                              [0.0299566, 0.184309, 1.46709]])

# Apply RGB to LMS transformation
lms_array = np.dot(original_array, rgb_to_lms_matrix.T)

# Color blindness simulation matrices
protanopia_matrix = np.array([[0, 2.02344, -2.52581],
                              [0, 1, 0],
                              [0, 0, 1]])

deuteranopia_matrix = np.array([[ 1.42319, -0.88995, 1.77557],
                                [ 0.67558, -0.42203, 2.82788],
                                [0.00267, -0.00504, 0.99914]])

tritanopia_matrix = np.array([[0.95451, -0.04719, 2.74872],
                              [-0.00447, 0.96543, 0.88835],
                              [-0.01251, 0.07312, -0.01161]])

# Simulate color blindness (equations M5, M6, M7)
simulated_lms_protanopia = np.dot(lms_array, protanopia_matrix.T)
simulated_lms_deuteranopia = np.dot(lms_array, deuteranopia_matrix.T)
simulated_lms_tritanopia = np.dot(lms_array, tritanopia_matrix.T)

# Inverse transformation to RGB (equation M8)
lms_to_rgb_matrix = np.linalg.inv(rgb_to_lms_matrix)
simulated_rgb_protanopia = np.dot(simulated_lms_protanopia, lms_to_rgb_matrix.T)
simulated_rgb_deuteranopia = np.dot(simulated_lms_deuteranopia, lms_to_rgb_matrix.T)
simulated_rgb_tritanopia = np.dot(simulated_lms_tritanopia, lms_to_rgb_matrix.T)

# Compute the difference between original and simulated RGB images (equation M9)
difference_protanopia = original_array - simulated_rgb_protanopia
difference_deuteranopia = original_array - simulated_rgb_deuteranopia
difference_tritanopia = original_array - simulated_rgb_tritanopia


protanopia_shift = np.array([[0, 0, 0],
                             [0.5, 1, 0],
                             [0.5, 0, 1]])
deuteranopia_shift = np.array([[1, 0.5, 0],
                               [0, 0, 0],
                               [0, 0.5, 1]])
tritanopia_shift = np.array([[1, 0, 0.7],
                             [0, 1, 0.7],
                             [0, 0, 0]])

shifted_protanopia = np.dot(difference_protanopia, protanopia_shift.T)
shifted_deuteranopia = np.dot(difference_deuteranopia, deuteranopia_shift.T)
shifted_tritanopia = np.dot(difference_tritanopia, tritanopia_shift.T)

compensated_rgb_protanopia = simulated_rgb_protanopia + shifted_protanopia
compensated_rgb_deuteranopia = simulated_rgb_deuteranopia + shifted_deuteranopia
compensated_rgb_tritanopia = simulated_rgb_tritanopia + shifted_tritanopia

compensated_rgb_protanopia = np.clip(compensated_rgb_protanopia, 0, 255).astype(np.uint8)
compensated_rgb_deuteranopia = np.clip(compensated_rgb_deuteranopia, 0, 255).astype(np.uint8)
compensated_rgb_tritanopia = np.clip(compensated_rgb_tritanopia, 0, 255).astype(np.uint8)

simulated_rgb_protanopia = np.clip(simulated_rgb_protanopia, 0, 255).astype(np.uint8)
simulated_rgb_deuteranopia = np.clip(simulated_rgb_deuteranopia, 0, 255).astype(np.uint8)
simulated_rgb_tritanopia = np.clip(simulated_rgb_tritanopia, 0, 255).astype(np.uint8)

# Convert NumPy arrays back to PIL images
result_image_protanopia = Image.fromarray(compensated_rgb_protanopia)
result_image_deuteranopia = Image.fromarray(compensated_rgb_deuteranopia)
result_image_tritanopia = Image.fromarray(compensated_rgb_tritanopia)

result_simulated_protanopia = Image.fromarray(simulated_rgb_protanopia)
result_simulated_deuteranopia = Image.fromarray(simulated_rgb_deuteranopia)
result_simulated_tritanopia = Image.fromarray(simulated_rgb_tritanopia)

# Display or save the results as needed
result_image_protanopia.show()
result_image_deuteranopia.show()
result_image_tritanopia.show()

# Save the results to files if needed
result_image_protanopia.save(common_path + "Result_Compensated_Protanopia.jpeg")
result_image_deuteranopia.save(common_path + "Result_Compensated_Deuteranopia.jpeg")
result_image_tritanopia.save(common_path + "Result_Compensated_Tritanopia.jpeg")

result_simulated_protanopia.save(common_path + "result_simulated_protanopia.jpeg")
result_simulated_deuteranopia.save(common_path + "result_simulated_deuteranopia.jpeg")
result_simulated_tritanopia.save(common_path + "result_simulated_tritanopia.jpeg")



