from collections import namedtuple
from typing import List, Tuple
from extract_digits import Digit
from constants import METERS_PER_PIXEL_HORIZONTAL, METERS_PER_PIXEL_VERTICAL


ImagePoint = namedtuple('ImagePoint', ['horizontal_pixels', 'vertical_pixels'])
DrawDigit = namedtuple('DrawDigit', ['horizontal', 'vertical', 'digit'])


def inches_to_meters(inches: float) -> float:
    return inches * 0.0254


def image_point_to_draw_digit(image_point: ImagePoint, digit_value: int, image_width: int, vertical_starting_point: float) -> DrawDigit:
    """
    Converts the pixel coordinates to meter coordinates where (0, 0) is the top center of the image frame.
    Positive horizontal values mean going right, and positive vertical values mean going down.

    Args:
        image_point: The (x, y) coordinates in pixels to convert
        digit_value: The value of the digit to draw
        image_width: The width (in pixels) of the entire image frame
        vertical_starting_point: The number of meters needed to move the arm to the top of the image frame
    Returns:
        A tuple containing the digit value, as well as the horizontal and vertical locations IN METERS of the digit to draw.
    """
    horizontal = (image_point.horizontal_pixels - int(image_width / 2)) * METERS_PER_PIXEL_HORIZONTAL
    vertical = image_point.vertical_pixels * METERS_PER_PIXEL_VERTICAL + vertical_starting_point

    return DrawDigit(horizontal=horizontal,
                     vertical=vertical,
                     digit=digit_value)

def add_numbers(top_digits: List[Digit], bottom_digits: List[Digit]) -> Tuple[List[int], List[int]]:
    top_values = list(reversed([d.value for d in top_digits]))
    bottom_values = list(reversed([d.value for d in bottom_digits]))

    while len(bottom_values) < len(top_values):
        bottom_values.append(0)

    result_digits: List[int] = []
    carry_digits: List[int] = []

    carry = 0
    for top, bottom in zip(top_values, bottom_values):
        digit_sum = top + bottom + carry
        digit = int(digit_sum % 10)
        carry = int(digit_sum / 10)

        result_digits.append(digit)
        carry_digits.append(carry)

    if carry > 0:
        result_digits.append(carry)
        carry_digits.append(0)

    return result_digits, carry_digits


def get_answer_locations(top_digits: List[Digit], bottom_digits: List[Digit], num_digits: int, draw_width: float) -> List[ImagePoint]:
    assert len(top_digits) >= len(bottom_digits), 'Must have at least as many top digits as bottom digits'

    results: List[ImagePoint] = []
    max_count = max(len(top_digits), len(bottom_digits))

    x_values: List[int] = []
    y_values: List[int] = []
    
    for idx in range(1, max_count + 1):
        top_idx = len(top_digits) - idx
        bottom_idx = len(bottom_digits) - idx

        if bottom_idx < 0:
            top_box = top_digits[top_idx].bounding_box
            draw_y = int(top_box.y + 3.5 * top_box.height)
            draw_x = int(top_box.x)
        else:
            bottom_box = bottom_digits[bottom_idx].bounding_box
            draw_y = int(bottom_box.y + 2.0 * bottom_box.height)
            draw_x = int(bottom_box.x)

        y_values.append(draw_y)
        x_values.append(draw_x)

    if (len(x_values) == 0) or len(y_values) == 0:
        return []

    min_vertical = min(y_values)

    for x_value in x_values:
        results.append(ImagePoint(horizontal_pixels=x_value, vertical_pixels=min_vertical))

    if len(results) == 0:
        return results

    while len(results) < num_digits:
        prev_x = results[-1].horizontal_pixels
        new_x = int(prev_x - (draw_width * 2.0) / METERS_PER_PIXEL_HORIZONTAL)
        results.append(ImagePoint(horizontal_pixels=new_x, vertical_pixels=min_vertical))

    return results
