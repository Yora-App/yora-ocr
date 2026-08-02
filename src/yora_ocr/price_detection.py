import re
import numpy as np
import shapely
import cv2

from collections.abc import Iterable


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector
    (i.e. makes it of length 1 while keeping its direction)
    """
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def centroid_of_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    Calculates mean of all points of a given polygon (centroid)
    """
    return np.mean(polygon, axis=0)


def baseline_vector_of_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    Calculates a normalized vector that points to the right of the polygon
    We assume that sideways is the direction in which the polygon has the longest edge
    (i.e. we assume that text is wider than high)
    """
    assert polygon.shape == (4, 2)

    # geometry operations are naturally float operations. Also int16 can overflow quickly
    polygon = polygon.astype(np.float64)

    longest_edge_i = 0
    longest_edge_length = -np.inf
    for i in range(4):
        length = np.linalg.norm(polygon[(i + 1) % 4] - polygon[i])
        if length > longest_edge_length:
            longest_edge_i = i
            longest_edge_length = length

    second_longest_edge_i = (longest_edge_i + 2) % 4  # 4 points in polygon
    longest_vector = polygon[(longest_edge_i + 1) % 4] - polygon[longest_edge_i]
    second_longest_vector = (
        polygon[(second_longest_edge_i + 1) % 4] - polygon[second_longest_edge_i]
    )

    # make sure both side edges point in the same direction before averaging
    if np.dot(longest_vector, second_longest_vector) < 0:
        second_longest_vector *= -1

    average_sideways_vector_normalized = normalize_vector(
        longest_vector + second_longest_vector
    )

    # make sure vector is pointing to the right
    if average_sideways_vector_normalized[0] < 0:
        average_sideways_vector_normalized *= -1

    return average_sideways_vector_normalized


def normal_vector_of_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    Calculates a normalized vector that points downwards from the polygon
    """
    baseline = baseline_vector_of_polygon(polygon)
    return np.array([-baseline[1], baseline[0]])


def draw_debug_overlay(
    image_path: str,
    polygon_stacks: list[list[np.ndarray]],
    polygon_stacks_prices: list[list[int]],
    non_price_polygons: list[np.ndarray],
    output_path: str,
):
    DEBUG_COLORS = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),
        (255, 128, 0),
    ]

    image = cv2.imread(image_path)
    assert image is not None

    for i, (stack, stack_prices) in enumerate(
        zip(polygon_stacks, polygon_stacks_prices)
    ):
        for polygon, price in zip(stack, stack_prices):
            # polygon: shape (4, 2)
            pts = polygon.astype(np.int32).reshape((-1, 1, 2))

            # draw OCR polygon
            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=DEBUG_COLORS[i % len(DEBUG_COLORS)],
                thickness=2,
            )

            # centroid
            centroid = centroid_of_polygon(polygon).astype(int)

            cv2.circle(
                image,
                tuple(centroid),
                radius=4,
                color=(0, 0, 255),
                thickness=-1,
            )

            # normal
            normal = normal_vector_of_polygon(polygon)

            end = centroid + normal * 100

            cv2.arrowedLine(
                image,
                tuple(centroid),
                tuple(end.astype(int)),
                color=(255, 0, 0),
                thickness=2,
                tipLength=0.2,
            )

            # price text
            cv2.putText(
                image,
                f"{price / 100:.2f}",
                tuple(centroid + np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    for polygon in non_price_polygons:
        # polygon: shape (4, 2)
        pts = polygon.astype(np.int32).reshape((-1, 1, 2))

        # draw OCR polygon
        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=(255, 255, 255),
            thickness=2,
        )

    cv2.imwrite(output_path, image)
    print(f"Wrote debug image to {output_path}")


def group_polygons_into_stacks(
    price_polygons: list[np.ndarray],
    price_polygon_prices: list[int],
    non_price_polygons: list[np.ndarray],
) -> tuple[list[list[np.ndarray]], list[list[int]]]:
    """
    Find polygon stacks (polygons stacked above each other without non_price_polygons in between)
    Returns all polygon stacks, each ordered from top to bottom.
    Also returns the price labels for each polygon in the same order
    """
    # calculate extend. We don't need to consider values larger than this for ray length etc
    extent = np.max(price_polygons) - np.min(price_polygons)

    shapely_price_polygons = shapely.polygons(price_polygons)
    shapely_non_price_polygons = shapely.polygons(non_price_polygons)
    assert isinstance(shapely_price_polygons, Iterable)
    assert isinstance(shapely_non_price_polygons, Iterable)

    # tuple of indices: first int points to previous polygon in stack, second to next polygon in stack
    polygon_pointers = [(-1, -1)] * len(price_polygons)
    for a in range(len(shapely_price_polygons)):
        centroid_a = centroid_of_polygon(price_polygons[a])
        centroid_a_shapely = shapely.Point(centroid_a)
        normal_a = normal_vector_of_polygon(price_polygons[a])
        normal_a_ray = shapely.LineString([centroid_a, centroid_a + extent * normal_a])

        # first we check distance to all non_price_polygons
        min_distance = extent
        for non_price_polygon in shapely_non_price_polygons:
            intersection = normal_a_ray.intersection(non_price_polygon)
            if not intersection.is_empty:
                distance = intersection.distance(centroid_a_shapely)
                if distance < min_distance:
                    min_distance = distance

        # next we check the distance to all price polygons
        # this distance must be smaller than the smallest non_price_polygons distance
        for b, polygon_b in enumerate(shapely_price_polygons):
            if b == a or b == polygon_pointers[a][0]:
                # omit current polygon and previous polygon
                continue
            intersection = normal_a_ray.intersection(polygon_b)
            if not intersection.is_empty:
                distance = intersection.distance(centroid_a_shapely)
                if distance < min_distance:
                    min_distance = distance
                    polygon_pointers[a] = (polygon_pointers[a][0], b)
                    polygon_pointers[b] = (a, polygon_pointers[b][1])

    # finally we have to convert our list of indices to a list of stacks
    stacks = []
    stacks_prices = []
    for i, (previous_i, next_i) in enumerate(polygon_pointers):
        if previous_i == -1:  # root of stack
            visited = {i}
            current_stack = [price_polygons[i]]
            current_stack_prices = [price_polygon_prices[i]]
            while next_i != -1:
                visited.add(next_i)
                current_stack.append(price_polygons[next_i])
                current_stack_prices.append(price_polygon_prices[i])
                i = next_i
                next_i = polygon_pointers[i][1]
            stacks.append(current_stack)
            stacks_prices.append(current_stack_prices)

    return stacks, stacks_prices


def find_item_prices_and_total_price(
    polygons: list[np.ndarray],
    ocr_texts: list[str],
    image_path: str,
    debug_image_output_path: str,
) -> tuple[list[np.ndarray], list[int], int]:
    price_polygons = []
    non_price_polygons = []
    price_polygons_prices = []

    # collect price polygons using a regex expression
    # this will catch too much though. Needs filtering through layout analysis
    for polygon, text in zip(polygons, ocr_texts):
        if match := re.search(r"(?<!\d)-?\d+[.,]\d{2}(?!\d)", text.strip()):
            price_polygons.append(polygon)
            price_polygons_prices.append(
                int(match[0].replace(",", "").replace(".", ""))
            )
        else:
            non_price_polygons.append(polygon)

    polygon_stacks, polygon_stacks_prices = group_polygons_into_stacks(
        price_polygons, price_polygons_prices, non_price_polygons
    )
    draw_debug_overlay(
        image_path,
        polygon_stacks,
        polygon_stacks_prices,
        non_price_polygons,
        debug_image_output_path,
    )

    largest_stack_size = len(polygon_stacks[0])
    largest_stack = polygon_stacks[0]
    largest_stack_prices = polygon_stacks_prices[0]
    for stack, stack_prices in zip(polygon_stacks, polygon_stacks_prices):
        if len(stack) > largest_stack_size:
            largest_stack = stack
            largest_stack_prices = stack_prices

    # filter out total price
    # we start by checking whether the last couple of elements are the same,
    # since the total price could be multiple times at the end of the stack
    total_index = len(largest_stack_prices) - 1
    while (
        total_index > 0
        and largest_stack_prices[total_index] == largest_stack_prices[total_index - 1]
    ):
        total_index -= 1

    total = 0
    for i in range(total_index):
        total += largest_stack_prices[i]

    if total == largest_stack_prices[total_index]:
        largest_stack = largest_stack[:total_index]
        largest_stack_prices = largest_stack_prices[:total_index]
    else:
        for i in range(total_index, len(largest_stack_prices)):
            total += largest_stack_prices[i]

    return largest_stack, largest_stack_prices, total
