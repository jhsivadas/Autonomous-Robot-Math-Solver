import cv2
import numpy as np
from collections import namedtuple
from sklearn.cluster import KMeans
from typing import Any, List, Tuple


MIN_HEIGHT = 35
MIN_AREA = 25
MERGE_DISTANCE = 10
Digit = namedtuple('Digit', ['value', 'image', 'bounding_box'])


class BoundingBox:

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def endpoints(self) -> np.ndarray:
        return np.array([[self.x, self.y], [self.x + self.width, self.y], [self.x, self.y + self.height], [self.x + self.width, self.y + self.height]])

    @property
    def area(self) -> int:
        return int(self.width * self.height)

    def distance_to(self, other: Any) -> int:
        if not isinstance(other, BoundingBox):
            return 1e10

        distances = np.sum(np.abs(np.expand_dims(self.endpoints, axis=0) - np.expand_dims(other.endpoints, axis=1)), axis=-1)  # [4, 4]
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
        self.width = (max_x - min_x)
        self.height = (max_y - min_y)


def group_digits(bounding_boxes: List[BoundingBox]) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    y_values = list(map(lambda b: b.y, bounding_boxes))
    clf = KMeans(n_clusters=2)  # Assume there are only 2 numbers
    clusters = clf.fit_predict(X=np.expand_dims(y_values, axis=-1), y=None)

    group_one: List[BoundingBox] = []
    group_two: List[BoundingBox] = []

    for cluster, box in zip(clusters, bounding_boxes):
        if cluster == 0:
            group_one.append(box)
        else:
            group_two.append(box)

    if min(map(lambda b: b.y, group_one)) < min(map(lambda b: b.y, group_two)):
        return group_one, group_two
    else:
        return group_two, group_one


def clip_to_bounding_box(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    return image[box.y:box.y + box.height, box.x:box.x + box.width]


def extract_digits(image: np.ndarray) -> List[BoundingBox]:
    # Convert image to grayscale and invert colors (the background is white)
    gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the image to extract the pen strokes
    threshold = cv2.THRESH_BINARY
    thresholded = cv2.threshold(gray, 160, 255, threshold)[1]

    # Get the contours from the thresholded image
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_rectangles = list(map(cv2.boundingRect, contours))
    bounding_boxes = [BoundingBox(x=x, y=y, width=w, height=h) for (x, y, w, h) in bounding_rectangles]

    if len(bounding_boxes) == 0:
        return []

    merged_bounding_boxes: List[BoundingBox] = []

    for box in bounding_boxes:
        # Skip very small boxes
        if (box.area < MIN_AREA):
            continue

        if len(merged_bounding_boxes) == 0:
            merged_bounding_boxes.append(box)
        else:
            distances = [box.distance_to(merged) for merged in merged_bounding_boxes]
            min_dist_idx = np.argmin(distances)

            if distances[min_dist_idx] < MERGE_DISTANCE:
                merged_bounding_boxes[min_dist_idx].merge(box)
            else:
                merged_bounding_boxes.append(box)

    digit_bounding_boxes: List[BoundingBox] = []

    for box in merged_bounding_boxes:
        if (box.area < MIN_AREA) or (box.height < MIN_HEIGHT):
            continue

        digit_bounding_boxes.append(box)

    # Group the digits in horizontal rows
    top_number, bottom_number = group_digits(digit_bounding_boxes)

    # Sort the bounding boxes by 'x' to get the digits in order (most significant to least significant)
    top_number = list(sorted(top_number, key=lambda b: b.x))
    bottom_number = list(sorted(bottom_number, key=lambda b: b.x))

    top_number_digits: List[Digit] = []
    bottom_number_digits: List[Digit] = []

    print('Top')

    for box in top_number:
        digit_img = clip_to_bounding_box(thresholded, box=box)
        digit = Digit(value=-1, image=digit_img, bounding_box=box)  # TODO: Classify the image to get the actual digit (both here and down below)
        top_number_digits.append(digit)

        cv2.imshow('Digit', digit_img)
        cv2.waitKey(0)

    print('Bottom')

    for box in bottom_number:
        digit_img = clip_to_bounding_box(thresholded, box=box)
        digit = Digit(value=-1, image=digit_img, bounding_box=box)
        bottom_number_digits.append(digit)

        cv2.imshow('Digit', digit_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    path = '../images/addition_problem.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img, (600, 600))

    cv2.imshow('Problem', img)
    cv2.waitKey(0)

    extract_digits(img)
