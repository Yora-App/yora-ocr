import re
import numpy as np
import shapely

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


def find_largest_polygon_stack(
    price_polygons: list[np.ndarray], non_price_polygons: list[np.ndarray]
) -> list[int]:
    """
    Finds price polygons that are stacked above each other (without non_price_polygons in between).
    Returns the largest of such stacks among the given polygons.
    Returns this stack as a list of indices of the polygons in this stack, ordered from top to bottom.
    """
    # calculate extend. We don't need to consider values larger than this for ray length etc
    extent = np.max(price_polygons) - np.min(price_polygons)

    shapely_price_polygons = shapely.polygons(price_polygons)
    shapely_non_price_polygons = shapely.polygons(non_price_polygons)
    assert isinstance(shapely_price_polygons, Iterable)
    assert isinstance(shapely_non_price_polygons, Iterable)

    # list of tuples: first element is pointer to next polygon in stack, second element is distance to it
    next_polygon_pointers = [(-1, extent)] * len(price_polygons)
    for a in range(len(shapely_price_polygons)):
        centroid_a = centroid_of_polygon(price_polygons[a])
        centroid_a_shapely = shapely.Point(centroid_a)
        normal_a = normal_vector_of_polygon(price_polygons[a])
        normal_a_ray = shapely.LineString([centroid_a, centroid_a + extent * normal_a])

        # first we check distance to all non_price_polygons
        for non_price_polygon in shapely_non_price_polygons:
            intersection = normal_a_ray.intersection(non_price_polygon)
            if not intersection.is_empty:
                distance = intersection.distance(centroid_a_shapely)
                if distance < next_polygon_pointers[a][1]:
                    next_polygon_pointers[a] = (-1, distance)

        # next we check the distance to all price polygons
        # this distance must be smaller than the smallest non_price_polygons distance
        for b, polygon_b in enumerate(shapely_price_polygons):
            if a == b:
                continue
            intersection = normal_a_ray.intersection(polygon_b)
            if not intersection.is_empty:
                distance = intersection.distance(centroid_a_shapely)
                if distance < next_polygon_pointers[a][1]:
                    next_polygon_pointers[a] = (b, distance)

    longest_stack = []
    longest_stack_length = 0
    for x in range(len(next_polygon_pointers)):
        i = x
        current_stack = [i]
        stack_length = 1
        next_i = next_polygon_pointers[i][0]
        visited = {i}
        while next_i != -1 and next_i not in visited:
            visited.add(next_i)
            stack_length += 1
            current_stack.append(next_i)
            i = next_i
            next_i = next_polygon_pointers[i][0]
        if stack_length > longest_stack_length:
            longest_stack = current_stack
            longest_stack_length = stack_length

    return longest_stack


def find_item_prices_and_total_price(
    polygons: list[np.ndarray], ocr_texts: list[str]
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

    filtered_polygon_indices = find_largest_polygon_stack(
        price_polygons, non_price_polygons
    )

    filtered_polygons = []
    filtered_polygons_prices = []
    for i in filtered_polygon_indices:
        filtered_polygons.append(price_polygons[i])
        filtered_polygons_prices.append(price_polygons_prices[i])

    # filter out total price
    # we start by checking whether the last couple of elements are the same,
    # since the total price could be multiple times at the end of the stack
    total_index = len(filtered_polygons_prices) - 1
    while (
        total_index > 0
        and filtered_polygons_prices[total_index]
        == filtered_polygons_prices[total_index - 1]
    ):
        total_index -= 1

    total = 0
    for i in range(total_index):
        total += filtered_polygons_prices[i]

    if total == filtered_polygons_prices[total_index]:
        filtered_polygons = filtered_polygons[:total_index]
        filtered_polygons_prices = filtered_polygons_prices[:total_index]
    else:
        for i in range(total_index, len(filtered_polygons_prices)):
            total += filtered_polygons_prices[i]

    return filtered_polygons, filtered_polygons_prices, total
