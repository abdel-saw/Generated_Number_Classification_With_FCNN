import numpy as np

def base_digits():
    """
    Defines base representations of digits (0-9) in a 6x12 matrix.
    1s represent filled parts, 0s represent empty parts.
    """
    digits = {
        '0': [
            " ****  ",
            "*    * ",
            "*    * ",
            "*    * ",
            "*    * ",
            " ****  "
        ],
        '1': [
            "   *   ",
            "  **   ",
            " * *   ",
            "   *   ",
            "   *   ",
            " ***** "
        ],
        '2': [
            " ****  ",
            "*    * ",
            "     * ",
            "   **  ",
            "  *    ",
            "****** "
        ],
        '3': [
            " ****  ",
            "     * ",
            "  ***  ",
            "     * ",
            "*    * ",
            " ****  "
        ],
        '4': [
            "    *  ",
            "   **  ",
            "  * *  ",
            " *  *  ",
            "****** ",
            "    *  "
        ],
        '5': [
            "****** ",
            "*      ",
            "*****  ",
            "     * ",
            "*    * ",
            " ****  "
        ],
        '6': [
            " ****  ",
            "*      ",
            "*****  ",
            "*    * ",
            "*    * ",
            " ****  "
        ],
        '7': [
            "****** ",
            "    *  ",
            "   *   ",
            "  *    ",
            " *     ",
            "*      "
        ],
        '8': [
            " ****  ",
            "*    * ",
            " ****  ",
            "*    * ",
            "*    * ",
            " ****  "
        ],
        '9': [
            " ****  ",
            "*    * ",
            " ****  ",
            "     * ",
            "*    * ",
            " ****  "
        ]
    }
    
    # Convert text to binary (0s and 1s)
    for key in digits:
        matrix = np.zeros((6, 12), dtype=int)
        for row_idx, row in enumerate(digits[key]):
            for col_idx, char in enumerate(row):
                if char != " ":
                    matrix[row_idx, col_idx] = 1
        digits[key] = matrix  # Store as numpy arrays
    
    return digits

def add_noise(digit_matrix, noise_level=0.1):
    """
    Adds random noise to a digit matrix by flipping some pixels.
    :param digit_matrix: The original 6x12 matrix
    :param noise_level: Fraction of pixels to flip (0 to 1)
    :return: Noisy digit matrix
    """
    noisy_matrix = digit_matrix.copy()
    num_flips = int(noise_level * digit_matrix.size)
    
    for _ in range(num_flips):
        row, col = np.random.randint(0, 6), np.random.randint(0, 12)
        noisy_matrix[row, col] = 1 - noisy_matrix[row, col]  # Flip 0->1 or 1->0
    
    return noisy_matrix

def shift_digit(digit_matrix, shift_x=0, shift_y=0):
    """
    Shifts the digit within the 6x12 matrix.
    :param digit_matrix: The original matrix
    :param shift_x: Horizontal shift (-2 to 2 recommended)
    :param shift_y: Vertical shift (-2 to 2 recommended)
    :return: Shifted matrix
    """
    shifted_matrix = np.zeros((6, 12), dtype=int)
    
    for i in range(6):
        for j in range(12):
            new_i, new_j = i + shift_y, j + shift_x
            if 0 <= new_i < 6 and 0 <= new_j < 12:
                shifted_matrix[new_i, new_j] = digit_matrix[i, j]
    
    return shifted_matrix

def generate_variations(digit_matrix, num_variations=20):
    """
    Generates multiple variations for a given digit.
    :param digit_matrix: The original matrix
    :param num_variations: Number of variations to create
    :return: List of varied matrices
    """
    variations = [digit_matrix]
    
    for _ in range(num_variations - 1):
        noisy = add_noise(digit_matrix, noise_level=0.1)
        shifted = shift_digit(noisy, shift_x=np.random.randint(-2, 3), shift_y=np.random.randint(-2, 3))
        variations.append(shifted)
    
    return variations
