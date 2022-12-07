import numpy as np
from collections import namedtuple, Counter
from typing import List, Tuple
from extract_digits import Digit
from constants import METERS_PER_PIXEL_HORIZONTAL, METERS_PER_PIXEL_VERTICAL


ImagePoint = namedtuple("ImagePoint", ["horizontal_pixels", "vertical_pixels"])
DrawDigit = namedtuple("DrawDigit", ["horizontal", "vertical", "digit"])


def inches_to_meters(inches: float) -> float:
    return inches * 0.0254


def image_point_to_draw_digit(
    image_point: ImagePoint,
    digit_value: int,
    image_width: int,
    vertical_starting_point: float,
) -> DrawDigit:
    """
    Converts the pixel coordinates to meter coordinates where (0, 0) is the top
    center of the image frame. Positive horizontal values mean going right, and
    positive vertical values mean going down.

    Args:
        image_point: The (x, y) coordinates in pixels to convert
        digit_value: The value of the digit to draw
        image_width: The width (in pixels) of the entire image frame
        vertical_starting_point: The number of meters needed to move the arm to
        the top of the image frame
    Returns:
        A tuple containing the digit value, as well as the horizontal and
        vertical locations IN METERS of the digit to draw.
    """
    horizontal = (
        image_point.horizontal_pixels - int(image_width * (11 / 25.5))
    ) * -METERS_PER_PIXEL_HORIZONTAL
    vertical = -1 * image_point.vertical_pixels * METERS_PER_PIXEL_VERTICAL

    print(
        "Vertial Dist: {} -> {}, {}, Horizontal Dist: {} -> {}".format(
            image_point.vertical_pixels,
            vertical,
            vertical + vertical_starting_point,
            image_point.horizontal_pixels,
            horizontal,
        )
    )
    vertical += vertical_starting_point

    return DrawDigit(horizontal=horizontal, vertical=vertical, digit=digit_value)


def add_numbers(
    top_digits: List[Digit], bottom_digits: List[Digit]
) -> Tuple[List[int], List[int]]:
    top_values = list(reversed([d.value for d in top_digits]))
    bottom_values = list(reversed([d.value for d in bottom_digits]))

    # adds implied zeroes to the left of the bottom value when < top value
    while len(bottom_values) < len(top_values):
        bottom_values.append(0)

    result_digits: List[int] = []
    carry_digits: List[int] = [0]

    # adds up the result values digit by digit
    carry = 0
    for top, bottom in zip(top_values, bottom_values):
        digit_sum = top + bottom + carry
        digit = int(digit_sum % 10)
        carry = int(digit_sum / 10)

        result_digits.append(digit)
        carry_digits.append(carry)

    if carry > 0:
        result_digits.append(carry)

    return result_digits, carry_digits


def get_answer_locations(
    top_digits: List[Digit],
    bottom_digits: List[Digit],
    num_digits: int,
    draw_width: float,
) -> List[ImagePoint]:
    assert len(top_digits) >= len(
        bottom_digits
    ), "Must have at least as many top digits as bottom digits"

    results: List[ImagePoint] = []
    max_count = max(len(top_digits), len(bottom_digits))

    x_values: List[int] = []
    y_values: List[int] = []

    # sets drawing x and y values for each digit
    for idx in range(1, max_count + 1):
        top_idx = len(top_digits) - idx
        bottom_idx = len(bottom_digits) - idx

        if bottom_idx < 0:
            top_box = top_digits[top_idx].bounding_box
            draw_y = int(top_box.y + 3.5 * top_box.height)
            draw_x = int(top_box.x)
        else:
            top_box = top_digits[top_idx].bounding_box
            bottom_box = bottom_digits[bottom_idx].bounding_box
            draw_y = int(bottom_box.y + 2.0 * bottom_box.height)
            draw_x = int((top_box.x + bottom_box.x) / 2)

        y_values.append(draw_y)
        x_values.append(draw_x)

    if (len(x_values) == 0) or len(y_values) == 0:
        return []

    vertical_pos = int(np.median(y_values))

    for x_value in x_values:
        results.append(
            ImagePoint(horizontal_pixels=x_value, vertical_pixels=vertical_pos)
        )

    # adds carry drawing locations
    carry_locs = []
    for digit in reversed(top_digits):
        carry_locs.append(
            ImagePoint(
                horizontal_pixels=digit.bounding_box.x,
                vertical_pixels=digit.bounding_box.y
                - (1.5 * (draw_width / METERS_PER_PIXEL_VERTICAL)),
            )
        )

    if len(results) == 0:
        return results

    # adds remaining digits to the left side of the answer 1.5*draw_width to
    # the left of the previously leftmost digit
    while len(results) < num_digits:
        prev_x = results[-1].horizontal_pixels
        new_x = int(prev_x - (draw_width * 1.5) / METERS_PER_PIXEL_HORIZONTAL)
        results.append(
            ImagePoint(horizontal_pixels=new_x, vertical_pixels=vertical_pos)
        )

    return results, carry_locs


def consolidate_digits(digit_lists: List[List[Digit]]) -> List[Digit]:
    digit_counts: Counter = Counter()
    for digit_list in digit_lists:
        digit_counts[len(digit_list)] += 1

    num_digits = digit_counts.most_common(1)[0][0]
    result: List[Digit] = []

    # iterates through each digit index and adds to digit instances if the index
    # is less than the length of the digit_list
    for digit_idx in range(num_digits):
        digit_instances: List[Digit] = []

        for digit_list in digit_lists:
            if digit_idx < len(digit_list):
                digit_instances.append(digit_list[digit_idx])

        merged = merge_digit_instances(digit_instances)
        result.append(merged)

    return result


def merge_digit_instances(digit_instances: List[Digit]) -> Digit:
    if len(digit_instances) <= 0:
        return Digit(value=-1, image=None, bounding_box=None)

    # Get the most frequent digit in this batch
    digit_counter: Counter = Counter()
    for digit in digit_instances:
        digit_counter[digit.value] += 1

    majority_digit = digit_counter.most_common(1)[0][0]

    # Return the first instance of the majority digit
    for digit in digit_instances:
        if digit.value == majority_digit:
            return digit

    return digit_instances[0]
