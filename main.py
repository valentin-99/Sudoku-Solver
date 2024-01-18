import cv2
import numpy as np
import re
from imutils import contours as imutils_contours
import pytesseract

def preprocess_image(image):

    # Get grayscale image
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('grayscale image', grayscaleImage)

    height, width = image.shape[:2]
    minLen = min(height, width)
    blockSize = 0
    
    print('width x height =  {} x {}'.format(width, height))

    # blocksize MUST BE ODD
    blockSize = minLen // 2
    if blockSize % 2 == 0:
        blockSize = blockSize + 1

    cst = minLen // 30

    print('blockSize = {}'.format(blockSize))
    print('cst = {}'.format(cst))

    # User adaptive threshold using gaussian_c calculation for a better image readability
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    thresholdImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, cst)
    # cv2.imshow('threshold image', thresholdImage)

    # detect largest contour
    contours, _ = cv2.findContours(thresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # create blank mask
    mask = np.zeros_like(thresholdImage)
    
    # draw on the mask the largest contour for the sudoku grid
    cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), thickness=cv2.FILLED)
    # cv2.imshow('mask', mask)

    # combine the threshold image and the mask
    grid_image = cv2.bitwise_and(thresholdImage, mask)
    # cv2.imshow('threshold + mask', grid_image)

    inverse = cv2.bitwise_not(grid_image)
    cv2.imshow('inverse', inverse)

    return inverse, contours

def find_corners(image, contours):
    largest_contour = max(contours, key=cv2.contourArea)

    # Find corners of the largest contour
    # Contour's perimeter
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    # Contour's approximation
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    # TEST Draw the largest contour on a blank image
    # contour_image  = np.ones_like(image)
    # cv2.drawContours(contour_image, [largest_contour], 0, (255, 255, 255), thickness=cv2.FILLED)
    # cv2.imshow('Contour Image', contour_image)

    # corners_image  = np.zeros_like(image)
    # cv2.drawContours(corners_image, [corners], 0, (255, 255, 255), thickness=cv2.FILLED)
    # cv2.imshow('Corners Image', corners_image)

    return corners

def check_black_pixels_margins(image, margin_size=10):
    # Extract the image dimensions
    height, width = image.shape[:2]

    # Define the regions of interest (ROIs) for the margins
    top_margin = image[:margin_size, :]
    bottom_margin = image[height - margin_size:, :]
    left_margin = image[:, :margin_size]
    right_margin = image[:, width - margin_size:]

    # Check if any of the margins have white pixels
    has_white_top = np.any(top_margin == 0)
    has_white_bottom = np.any(bottom_margin == 0)
    has_white_left = np.any(left_margin == 0)
    has_white_right = np.any(right_margin == 0)

    # Return the results
    return has_white_top or has_white_bottom or has_white_left or has_white_right

def add_white_margins(image, margin_size=10):
    # Extract the image dimensions
    height, width = image.shape[:2]

    # Create a new image with increased dimensions
    new_height = height + 2 * margin_size
    new_width = width + 2 * margin_size
    new_image = np.ones((new_height, new_width), dtype=np.uint8) * 255  # Initialize with white pixels

    # Copy the original image to the center of the new image
    new_image[margin_size:margin_size + height, margin_size:margin_size + width] = image

    return new_image

# https://stackoverflow.com/questions/58985400/how-to-fix-transform-perspective-function-incorrectly-returning-image-in-the-wro
def is_top_right_higher_than_top_left(contours):
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the coordinates of the top-left and top-right corners
    top_left = (x, y)
    top_right = (x + w, y)

    # Check if the top-right corner is higher than the top-left corner
    return top_right[1] < top_left[1]

def wrap_image(image, contours):
    # Find corners
    corners = find_corners(image, contours)

    # additional white margins layer
    if (check_black_pixels_margins(image)):
        print("Black pixels on margins")
        image = add_white_margins(image)
    # cv2.imshow('Additional margins', image)

    temp = 10
    # Define the destination points based on the image size
    height, width = image.shape[:2]
    dest_points = np.float32([[temp, temp], 
                              [temp, height - temp], 
                              [width - temp, height - temp], 
                              [width - temp, temp]])

    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dest_points)

    # Apply the perspective transformation
    wrapped_image = cv2.warpPerspective(image, matrix, (width, height))

    # "rotate" -90 degrees to correct orientation
    if (is_top_right_higher_than_top_left):
        wrapped_image = cv2.transpose(wrapped_image)
        wrapped_image = cv2.flip(wrapped_image, 1)
        # cv2.imshow('Wrapped Image -90', wrapped_image)

    cv2.imshow('Wrapped Image', wrapped_image)
    # cv2.waitKey(0)

    return wrapped_image

def detect_grid_lines(image):
    # Get inverse image + adaptiveThreshold preprocessing
    thresholdImage = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # cv2.imshow('thresholdImage', thresholdImage)

    # find the lines using probabilistic Hough line transform
    # NOTE -> adjust maxLineGap, it determines the maximum gap between lines 
    # segments that can be connected to a line.
    height, width = image.shape[:2]
    minLen = min(height, width)
    gap = minLen // 9
    print('gap = {}'.format(gap))
    lines = cv2.HoughLinesP(thresholdImage, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=gap)

    # create a blank image to draw the lines
    lines_image = np.zeros_like(image)

    # draw the detected lines on the image
    # NOTE -> adjust the line thickness if needed
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lines_image, (x1, y1), (x2, y2), 255, 5)
    # cv2.imshow('Hough lines', lines_image)

    # get intersection between grid image and the lines
    intersections = cv2.bitwise_and(thresholdImage, lines_image)
    cv2.imshow('intersections image', intersections)

    # cv2.waitKey(0)

    return intersections

def extract_cells(image, intersections):
    # Find contours in the intersection points image
    contours, _ = cv2.findContours(intersections, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Remove the largest contour (outer contour of the entire grid)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]

    # Sort contours top-to-bottom
    contours, _ = imutils_contours.sort_contours(contours, method="top-to-bottom")

    # Split the contours into rows
    rows = []
    for i in range(0, len(contours), 9):
        row = []
        for j in range(9):
            if i + j < len(contours):
                row.append(contours[i + j])
        rows.append(row)

    # Calculate the size of each cell
    cell_size = image.shape[0] // 9

    # Initialize variables to store cells and cell contours
    cells = []
    cell_contours = []

    for row in rows:
        # Sort contours from left to right within each row
        row, _ = imutils_contours.sort_contours(row, method="left-to-right")

        for contour in row:
            x, y, w, h = cv2.boundingRect(contour)

            # Skip if the contour is too small or too big to be a cell
            if w > cell_size // 2 and h > cell_size // 2 and \
                    w < cell_size * 2 and h < cell_size * 2:
                cell = image[y:y + h, x:x + w]
                cells.append(cell)
                cell_contours.append(contour)

    # Print the cells
    # for i, cell in enumerate(cells):
    #     cv2.imshow(f"Cell {i + 1}", cell)
    #     print(f"Cell {i + 1}")
    #     cv2.waitKey(0)

    return cells, cell_contours, cell_size

def transform_margins_to_black(image, cell_size):
    # Get the image dimensions
    height, width = image.shape

    # Set the margin size
    margin_size = cell_size // 10

    # Transform the top margin to black
    image[0:margin_size, :] = 0

    # Transform the bottom margin to black
    image[height-margin_size:height, :] = 0

    # Transform the left margin to black
    image[:, 0:margin_size] = 0

    # Transform the right margin to black
    image[:, width-margin_size:width] = 0

    return image

def recognize_digits(cells, cell_size):
    sudoku_digits = []

    # for cell in cells:
    for cell in cells:
        blur = cv2.GaussianBlur(cell, (5, 5), 0)
        cell = transform_margins_to_black(blur, cell_size)

        # cv2.imshow('cell', cell)
        # cv2.waitKey(0)

        digit = pytesseract.image_to_string(cell, config='--psm 13 --oem 3 -c tessedit_char_whitelist=123456789')

        digit = re.sub(r'\D', '', digit.strip())

        # print(digit.isdigit())
        # print(check_white_pixels(cell, cell_size))
        # print("Recognized digit:", digit)

        if digit.isdigit():
            # convert the normal digit
            digit_int = int(digit)
            sudoku_digits.append(digit_int)
        else:
            # empty cell
            sudoku_digits.append(0)

    return sudoku_digits

# debug function
def draw_contours(image, contours):
    # Create a copy of the image to draw the contours on
    image_with_contours = image.copy()

    # Draw each contour with a random color
    for i, contour in enumerate(contours):
        # Draw the contour on the image
        cv2.drawContours(image_with_contours, [contour], -1, (155, 155, 155), thickness=2)

    return image_with_contours

def solve_sudoku(image):

    # Step 1.1: preprocess the input image
    preprocessed_image, contours = preprocess_image(image)
    
    # Step 1.2: wrap the preprocessed image
    wrapped_image = wrap_image(preprocessed_image, contours)

    # Step 2: detect the sudoku lines from the preprocessed image 
    intersections = detect_grid_lines(wrapped_image)

    # Step 3: extract all the cells from the preprocessed image
    cells, cell_contours, cell_size = extract_cells(wrapped_image, intersections)
    
    # Step 3.1: Throw error
    if len(cells) < 81:
        raise Exception("The image couldn't be processed - not enough cells detected - {}".format(len(cells)))
    elif len(cells) > 81:
        raise Exception("The image couldn't be processed - too many cells detected - {}".format(len(cells)))

    # DEBUG - Draw contours
    aux_img = draw_contours(wrapped_image, cell_contours)
    cv2.imshow('aux_img', aux_img)

    # Step 4: recognize the digits from the cells vector
    sudoku_digits = recognize_digits(cells, cell_size)

    # DEBUG - reshape array (dimension = 81) to a 9x9 matrix 
    grid = np.array(sudoku_digits).reshape((9, 9))
    print(np.matrix(grid))

def main():
    # Get the image path
    image_path = "sudoku1.png"
    
    # Read the image
    image = cv2.imread(image_path)
    # cv2.imshow('input image', image)

    # Solve sudoku from image
    solve_sudoku(image)

    cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
