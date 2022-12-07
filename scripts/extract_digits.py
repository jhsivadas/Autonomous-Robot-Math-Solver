import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from scipy.signal import correlate2d
from typing import Any, List, Tuple


MIN_HEIGHT = 10
MIN_AREA = 200
MERGE_DISTANCE = 10
MAX_RATIO = 2.0
Digit = namedtuple("Digit", ["value", "image", "bounding_box"])


class BoundingBox:
    """
    Object handling a bounding box in an image.
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def endpoints(self) -> np.ndarray:
        return np.array(
            [
                [self.x, self.y],
                [self.x + self.width, self.y],
                [self.x, self.y + self.height],
                [self.x + self.width, self.y + self.height],
            ]
        )

    @property
    def area(self) -> int:
        return int(self.width * self.height)

    def distance_to(self, other: Any) -> int:
        if not isinstance(other, BoundingBox):
            return 1e10

        distances = np.sum(
            np.abs(
                np.expand_dims(self.endpoints, axis=0)
                - np.expand_dims(other.endpoints, axis=1)
            ),
            axis=-1,
        )  # [4, 4]
        return int(np.min(distances))

    def merge(self, other: Any):
        if not isinstance(other, BoundingBox):
            return

        min_x = min(self.x, other.x)
        max_x = max(self.x + self.width, other.x + other.width)
        min_y = min(self.y, other.y)
        max_y = max(self.y + self.height, other.y + other.height)

        self.x = min_x
        self.y = min_y
        self.width = max_x - min_x
        self.height = max_y - min_y


class DigitClassifier:
    """
    A digit classifier implemented using a convolutional neural network. This classifier
    only handles the forward (inference) operation of the neural network.
    """

    def __init__(self, weights_path: str):
        # Unpack the trained model weights
        with h5py.File(weights_path, "r") as fin:
            self.conv0_filter = fin["model_weights"]["conv0"]["conv0"]["kernel:0"][:]
            self.conv0_bias = fin["model_weights"]["conv0"]["conv0"]["bias:0"][:]

            self.conv1_filter = fin["model_weights"]["conv1"]["conv1"]["kernel:0"][:]
            self.conv1_bias = fin["model_weights"]["conv1"]["conv1"]["bias:0"][:]

            self.dense_hidden_weights = fin["model_weights"]["dense_hidden"][
                "dense_hidden"
            ]["kernel:0"][:]
            self.dense_hidden_bias = fin["model_weights"]["dense_hidden"][
                "dense_hidden"
            ]["bias:0"][:]

            self.output_weights = fin["model_weights"]["output"]["output"]["kernel:0"][
                :
            ]
            self.output_bias = fin["model_weights"]["output"]["output"]["bias:0"][:]

    def conv2d(
        self,
        inputs: np.ndarray,
        conv_filters: np.ndarray,
        conv_bias: np.ndarray,
        stride: int,
    ) -> np.ndarray:
        """
        Performs a single 2d convolution with multiple input and output channels.

        Args:
            inputs: A [S, S, K] array of input features
            conv_filters: A [T, T, K, L] array of T x T convolution filters with K input channels and L output channels
            conv_bias: A [T', T', L] additive bias
            stride: The stride length for this filter
        Returns:
            A [T', T', L] array containing the convolution result.
        """
        # Get the number of input channels (K) and output channels (L)
        _, _, num_input_channels, num_output_channels = conv_filters.shape

        results: List[
            np.ndarray
        ] = []  # Will hold list of L [R, R, 1] arrays (one for each output channel)

        for output_channel_idx in range(num_output_channels):

            # Execute each input channel using the given filters and stride lengths
            input_channel_conv_list: List[np.ndarray] = []
            for input_channel_idx in range(num_input_channels):
                conv_multi_in = correlate2d(
                    inputs[:, :, input_channel_idx],
                    conv_filters[:, :, input_channel_idx, output_channel_idx],
                    mode="valid",
                )  # [S', S']

                if stride > 1:
                    conv_multi_in = conv_multi_in[::stride, ::stride]

                input_channel_conv_list.append(np.expand_dims(conv_multi_in, axis=-1))

            # Sum the results over the input channels
            input_channel_conv = np.concatenate(
                input_channel_conv_list, axis=-1
            )  # [R, R, K]
            conv_sums = np.sum(input_channel_conv, axis=-1, keepdims=True)  # [R, R, 1]
            results.append(conv_sums)  # [R, R, 1]

        # Concatenate the output channels
        conv_result = np.concatenate(results, axis=-1)  # [R, R, L]

        # Apply the bias and relu activation
        linear_transformed = conv_result + np.reshape(
            conv_bias, (1, 1, -1)
        )  # [R, R, L]
        return np.maximum(linear_transformed, 0)

    def dense(
        self,
        inputs: np.ndarray,
        weight_matrix: np.ndarray,
        bias: np.ndarray,
        should_activate: bool,
    ) -> np.ndarray:
        """
        Executes a single dense neural network layer.

        Args:
            inputs: A [K] vector of input features
            weight_matrix: A [K, M] matrix of trained weights
            bias: A [M] matrix containing the trained bias
            should_activate: Whether to apply a ReLU activation
        Returns:
            A vector of size [M] containing the transformed values
        """
        linear_transformed = np.matmul(inputs, weight_matrix) + bias

        if should_activate:
            return np.maximum(linear_transformed, 0)

        return linear_transformed

    def predict(self, image: np.ndarray) -> int:
        """
        Performs a digit classification operations on a 2d unnormalized image

        Args:
            image: A [H, W] image containing pixel values in [0, 255]
        Returns:
            The model's digit prediction (0-9)
        """

        # First, re-size the image to 28 x 28. To aid in the digit classification,
        # we pad the edges to place the digit in the middle of the frame
        resized = np.zeros(shape=(28, 28), dtype=float)
        resized[2:26, 2:26] = cv2.resize(image, (24, 24)).astype(float)
        resized = np.expand_dims(resized, axis=-1)  # [28, 28, 1]

        # Normalize the pixel values in the range [0, 1]
        resized /= 255.0

        # Apply the convolution filters
        conv0 = self.conv2d(
            resized, conv_filters=self.conv0_filter, conv_bias=self.conv0_bias, stride=2
        )
        conv1 = self.conv2d(
            conv0, conv_filters=self.conv1_filter, conv_bias=self.conv1_bias, stride=2
        )

        # Apply the dense layers
        flattened = conv1.reshape(-1)
        dense_hidden = self.dense(
            inputs=flattened,
            weight_matrix=self.dense_hidden_weights,
            bias=self.dense_hidden_bias,
            should_activate=True,
        )
        logits = self.dense(
            inputs=dense_hidden,
            weight_matrix=self.output_weights,
            bias=self.output_bias,
            should_activate=False,
        )

        return np.argmax(logits)


def group_digits(
    bounding_boxes: List[BoundingBox],
) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    """
    Groups digits into 2 horizontal rows as would be seen in an addition problem. This grouping works by
    finding a vertical threshold to split the observed digit bounding boxes
    """
    y_values = list(map(lambda b: b.y, bounding_boxes))

    # Find the largest gap in y values
    sorted_y_values = list(sorted(y_values))
    gaps = [
        sorted_y_values[i] - sorted_y_values[i - 1]
        for i in range(1, len(sorted_y_values))
    ]
    max_gap_idx = np.argmax(gaps)

    # Set the split point based on the largest gap and cluster the digits
    split_point_y = (
        sorted_y_values[max_gap_idx] + sorted_y_values[max_gap_idx + 1]
    ) / 2.0
    clusters = [int(box.y >= split_point_y) for box in bounding_boxes]

    group_one: List[BoundingBox] = []
    group_two: List[BoundingBox] = []

    for cluster, box in zip(clusters, bounding_boxes):
        if cluster == 0:
            group_one.append(box)
        else:
            group_two.append(box)

    # Always return the top row first
    if min(map(lambda b: b.y, group_one)) < min(map(lambda b: b.y, group_two)):
        return group_one, group_two
    else:
        return group_two, group_one


def clip_to_bounding_box(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    return image[box.y : box.y + box.height, box.x : box.x + box.width]


def extract_digits(
    image: np.ndarray, digit_classifier: DigitClassifier
) -> Tuple[List[Digit], List[Digit]]:
    """
    Extracts digits from the given image and classifies them using the given classifier. This function
    returnts the top and bottom digits in order of most significant to least significant.
    """
    # Convert image to HSV colors and extract the blue lines (which are the pen)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresholded = cv2.inRange(hsv_img, (80, 75, 75), (120, 255, 255))

    # Get the contours from the thresholded image
    contours, hierarchy = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the bounding boxes for each contour
    bounding_rectangles = list(map(cv2.boundingRect, contours))
    bounding_boxes = [
        BoundingBox(x=x, y=y, width=w, height=h) for (x, y, w, h) in bounding_rectangles
    ]

    if len(bounding_boxes) == 0:
        return [], []

    digit_bounding_boxes: List[BoundingBox] = []

    for box in bounding_boxes:
        # Filter out boxes that are very wide but short. This generally is the line at the bottom
        # of the arithmetic problem.
        if (box.width / box.height) >= MAX_RATIO:
            continue

        digit_bounding_boxes.append(box)

    # Merge adjacent boxes that are small. This handles some inconsistencies in
    # the color extraction (we are working with a low resolution camera)
    merged_bounding_boxes: List[BoundingBox] = []
    for box in digit_bounding_boxes:
        min_dists = [b.distance_to(box) for b in merged_bounding_boxes]

        if len(min_dists) == 0:
            merged_bounding_boxes.append(box)
        else:
            min_idx = np.argmin(min_dists)
            if (min_dists[min_idx] < MERGE_DISTANCE) and (box.area < MIN_AREA):
                merged_bounding_boxes[min_idx].merge(box)
            else:
                merged_bounding_boxes.append(box)

    # Split boxes that have an overly large height. This often happens when the vertical contours get merged.
    box_heights = list(map(lambda b: b.height, merged_bounding_boxes))
    height_threshold = np.median(box_heights) + 2.0 * (
        np.percentile(box_heights, 75) - np.percentile(box_heights, 25)
    )

    split_bounding_boxes: List[BoundingBox] = []
    for box in merged_bounding_boxes:
        # Filter out boxes that are too small, as these are likely not digits. The exception is `1`, which will
        # be a small vertical line sometimes. We explicitly handle this edge case by looking at the ratio between height and width.
        if (box.area < MIN_AREA) and ((box.height / box.width) <= MAX_RATIO):
            continue

        if box.height > height_threshold:
            new_height = int(box.height / 2)
            split_bounding_boxes.append(
                BoundingBox(x=box.x, y=box.y, width=box.width, height=new_height)
            )
            split_bounding_boxes.append(
                BoundingBox(
                    x=box.x, y=box.y + new_height, width=box.width, height=new_height
                )
            )
        else:
            split_bounding_boxes.append(box)

    if len(split_bounding_boxes) == 0:
        return [], []

    # Group the digits in horizontal rows
    top_number, bottom_number = group_digits(split_bounding_boxes)

    # Sort the bounding boxes by 'x' to get the digits in order (most significant to least significant)
    top_number = list(sorted(top_number, key=lambda b: b.x))
    bottom_number = list(sorted(bottom_number, key=lambda b: b.x))

    top_number_digits: List[Digit] = []
    bottom_number_digits: List[Digit] = []

    #for box in split_bounding_boxes:
    #    cv2.rectangle(image, (box.x, box.y), (box.x + box.width, box.y + box.height), (0, 255, 0))

    #cv2.imshow('image', image)
    #cv2.waitKey(0)

    # Use grayscale thresholding to extract the digits. This gives a sharper image.
    #grayscale = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #_, gray_thresholded = cv2.threshold(
    #    grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    #)

    for box in top_number:
        # Clip the image and convert to grayscale
        digit_img_color = clip_to_bounding_box(image, box=box)

        digit_grayscale = 255 - cv2.cvtColor(digit_img_color, cv2.COLOR_BGR2GRAY)
        _, digit_img = cv2.threshold(
            digit_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        digit_value = digit_classifier.predict(digit_img)
        digit = Digit(value=digit_value, image=digit_img, bounding_box=box)
        top_number_digits.append(digit)

    for box in bottom_number:
        digit_img_color = clip_to_bounding_box(image, box=box)

        digit_grayscale = 255 - cv2.cvtColor(digit_img_color, cv2.COLOR_BGR2GRAY)
        _, digit_img = cv2.threshold(
            digit_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        digit_value = digit_classifier.predict(digit_img)
        digit = Digit(value=digit_value, image=digit_img, bounding_box=box)
        bottom_number_digits.append(digit)

    #cv2.imwrite(os.path.join(os.path.dirname(__file__), "image.png"), image)

    return top_number_digits, bottom_number_digits


if __name__ == "__main__":
    path = "../images/image.png"
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    cv2.imshow("Problem", img)
    cv2.waitKey(0)

    digit_classifier = DigitClassifier("mnist.h5")
    top_number_digits, bottom_number_digits = extract_digits(img, digit_classifier)

    print(
        "Top Number: {}".format(
            "".join(map(str, [digit.value for digit in top_number_digits]))
        )
    )
    print(
        "Bottom Number: {}".format(
            "".join(map(str, [digit.value for digit in bottom_number_digits]))
        )
    )
