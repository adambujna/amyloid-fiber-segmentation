import numpy as np
import shapely.geometry as sp
from typing import List, Tuple

# Types
Annotations = List[Tuple[int, List[Tuple[int, int]]]]
BoundingBox = Tuple[int, int, int, int]


def tile_image(image: np.ndarray, labels: Annotations = None,
               size: tuple[int, int] = (1024, 1024), min_overlap: int = 256,
               verbose: int = 0) -> tuple[list[tuple[np.ndarray, Annotations]], list[BoundingBox]]:
    """
    Tiles a given image and its annotation data into smaller sub-images.
    Original position information is returned for recreation.

    The number of windows required to cover the array is equal to N = ceil((A - W) / (W - O)) + 1,
    where A is array size, W is window size, and O is overlap size.

    The overlap to fit exactly without padding is then O = (A - NW) / (1 - N).
    """
    # Calculate overlap
    H, W = image.shape[:2]
    N_WINDOWS_X = np.ceil((W - size[1]) / (size[1] - min_overlap)) + 1
    N_WINDOWS_Y = np.ceil((H - size[0]) / (size[0] - min_overlap)) + 1
    OVERLAP_X = (W - N_WINDOWS_X * size[1]) / (1 - N_WINDOWS_X) if N_WINDOWS_X > 1 else 0
    OVERLAP_Y = (H - N_WINDOWS_Y * size[0]) / (1 - N_WINDOWS_Y) if N_WINDOWS_Y > 1 else 0
    # Calculate stride
    stride = (int(size[0] - OVERLAP_Y), int(size[1] - OVERLAP_X))

    if verbose > 0:
        print(f"Horizontal Windows: {N_WINDOWS_X}\nVertical Windows: {N_WINDOWS_Y}\n------------\n"
              f"Horizontal Overlap: {OVERLAP_X}\nVertical Overlap: {OVERLAP_Y}\n------------\n"
              f"Stride: X = {stride[1]}, Y = {stride[0]}")

    new_images_with_annotations = []
    boxes = []

    for y in range(0, H - size[0] + 1, stride[0]):
        for x in range(0, W - size[1] + 1, stride[1]):
            boxes.append((y, x, y + size[0], x + size[1]))

            tile = image[y:y + size[0], x:x + size[1]]
            new_annotations = []

            if not labels:
                new_images_with_annotations.append((tile, []))
                continue

            # Update each annotation to fit inside the new tile image
            for class_index, points in labels:
                # Get part of the annotation that is in tile
                if len(points) < 3:  # Line segmentation is invalid but can happen.
                    print(f"Invalid label of size ({len(points)}) found. Skipping.")
                    continue
                poly = sp.Polygon(points)
                if not poly.is_valid:  # In case that a point lies on a line between two other points
                    poly = poly.buffer(0)
                crop = sp.box(minx=x, miny=y, maxx=x + size[1], maxy=y + size[0])
                intersect = poly.intersection(crop)

                # Create a new annotation and add it to annotations for this image
                # Intersection can be either a polygon or multiple polygons, or is discarded
                if intersect.geom_type == 'Polygon':
                    if not intersect.is_empty:
                        new_contour = [(point_x - x, point_y - y) for point_x, point_y in intersect.exterior.coords]
                        new_annotations.append((class_index, new_contour))
                elif intersect.geom_type == 'MultiPolygon':
                    for cont in intersect.geoms:
                        if cont.is_empty:
                            continue  # Discard empty ones
                        new_contour = [(point_x - x, point_y - y) for point_x, point_y in cont.exterior.coords]
                        new_annotations.append((class_index, new_contour))

            # Put cropped tile and its annotations into output
            new_images_with_annotations.append((tile, new_annotations))
    # Return a list of (image, annotations) tuples and return positions and sizes of boxes
    return new_images_with_annotations, boxes


def shift_contours(cons: list[np.ndarray], x: int, y: int) -> list[np.ndarray]:
    """
    Shifts a contour as an array of points on X and Y axes.
    """
    shifted = cons[:]
    for c in shifted:
        c[:, 0] += x
        c[:, 1] += y
    return shifted


def polygon_iou(poly1, poly2):
    if len(poly1) == 0 or len(poly2) == 0:
        return 0.0
    polygon1, polygon2 = sp.Polygon(poly1), sp.Polygon(poly2)

    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area

    return intersection / union


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def merge_contours(pos1: tuple[int, int, int, int], pos2: tuple[int, int, int, int],
                   cons1: list[np.ndarray] = None, cons2: list[np.ndarray] = None,
                   iou_threshold: float = 0.5) -> list[np.ndarray]:
    if not cons1:
        return cons2 if cons2 else []
    if not cons2:
        return cons1

    # Determine global canvas dimensions
    y_min = min(pos1[0], pos2[0])
    x_min = min(pos1[1], pos2[1])

    y1_off, x1_off = pos1[0] - y_min, pos1[1] - x_min
    y2_off, x2_off = pos2[0] - y_min, pos2[1] - x_min

    # Shift all contours to global coordinates
    global_cons1 = shift_contours(cons1, x1_off, y1_off)
    global_cons2 = shift_contours(cons2, x2_off, y2_off)

    # Compute overlap region in global coordinates
    xo1 = max(pos1[1], pos2[1]) - x_min
    yo1 = max(pos1[0], pos2[0]) - y_min
    xo2 = min(pos1[3], pos2[3]) - x_min
    yo2 = min(pos1[2], pos2[2]) - y_min
    crop_box = sp.box(xo1, yo1, xo2, yo2)

    # Find intersecting polygons in the overlap area
    inters1, inters2 = [], []
    for i, contours in enumerate([global_cons1, global_cons2]):
        for object_id, c in enumerate(contours):
            poly = sp.Polygon(c).buffer(0)
            if poly.is_empty or not poly.is_valid:
                continue
            if not poly.intersects(crop_box):
                continue
            intersect = poly.intersection(crop_box)
            if intersect.is_empty:
                continue
            target = inters1 if i == 0 else inters2
            if intersect.geom_type == 'Polygon':
                target.append((object_id, intersect.exterior.coords))
            elif intersect.geom_type == 'MultiPolygon':
                for p in intersect.geoms:
                    if not p.is_empty:
                        target.append((object_id, p.exterior.coords))

    # Match and merge intersecting polygons
    matched_1 = set()
    matched_2 = set()
    final_contours = []

    for id1, poly1_coords in inters1:
        best_iou = 0
        best_match = None
        for id2, poly2_coords in inters2:
            if id2 in matched_2:
                continue
            iou = polygon_iou(poly1_coords, poly2_coords)
            if iou > best_iou:
                best_iou = iou
                best_match = id2
        if best_iou >= iou_threshold and best_match is not None:
            matched_1.add(id1)
            matched_2.add(best_match)
            poly1 = sp.Polygon(global_cons1[id1]).buffer(0)
            poly2 = sp.Polygon(global_cons2[best_match]).buffer(0)
            union = poly1.union(poly2)
            if union.geom_type == 'Polygon':
                final_contours.append(np.array(union.exterior.coords))
            elif union.geom_type == 'MultiPolygon':
                final_contours.append(np.array(union.convex_hull.exterior.coords))

    # Add unmatched contours
    for i, c in enumerate(global_cons1):
        if i not in matched_1:
            final_contours.append(c)
    for i, c in enumerate(global_cons2):
        if i not in matched_2:
            final_contours.append(c)

    return final_contours


def merge_masks(pos1: tuple[int, int, int, int], pos2: tuple[int, int, int, int],
                mask1_list: list[np.ndarray] = None, mask2_list: list[np.ndarray] = None,
                iou_threshold: float = 0.5) -> list[np.ndarray]:
    if not mask1_list:
        return mask2_list if mask2_list else []
    if not mask2_list:
        return mask1_list
    # Tile positions
    y11, x11, y12, x12 = pos1
    y21, x21, y22, x22 = pos2

    # Determine the size of global canvas
    y_min = min(y11, y21)
    x_min = min(x11, x21)
    y_max = max(y12, y22)
    x_max = max(x12, x22)
    height = y_max - y_min
    width = x_max - x_min

    # Offset for each tile
    y1_off, x1_off = pos1[0] - y_min, pos1[1] - x_min
    y2_off, x2_off = pos2[0] - y_min, pos2[1] - x_min

    # Overlap
    xo1, yo1, xo2, yo2 = max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)
    overlap = np.zeros((height, width), dtype=np.uint8)
    overlap[yo1:yo2 + 1, xo1:xo2 + 1] = 255

    # Position masks into global canvas
    global_masks1 = []
    for m in mask1_list:
        canvas = np.zeros((height, width), dtype=np.uint8)
        h, w = m.shape
        canvas[y1_off:y1_off + h, x1_off:x1_off + w] = m
        global_masks1.append(canvas)

    global_masks2 = []
    for m in mask2_list:
        canvas = np.zeros((height, width), dtype=np.uint8)
        h, w = m.shape
        canvas[y2_off:y2_off + h, x2_off:x2_off + w] = m
        global_masks2.append(canvas)

    # Compare masks in the overlap region
    matched_1 = set()
    matched_2 = set()
    merged_masks = []

    for i, m1 in enumerate(global_masks1):
        best_iou = 0
        best_j = None
        overlap1 = np.logical_and(m1, overlap)
        for j, m2 in enumerate(global_masks2):
            if j in matched_2:
                continue
            overlap2 = np.logical_and(m2, overlap)
            iou = mask_iou(overlap1, overlap2)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            matched_1.add(i)
            matched_2.add(best_j)
            merged = np.logical_or(global_masks1[i], global_masks2[best_j]).astype(np.uint8)
            merged_masks.append(merged)

    # Add unmatched masks
    for i, m in enumerate(global_masks1):
        if i not in matched_1:
            merged_masks.append(m)
    for j, m in enumerate(global_masks2):
        if j not in matched_2:
            merged_masks.append(m)

    return merged_masks


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image1 = np.zeros((1125, 1500), dtype=np.uint8)
    finals = merge_contours((0, 0, 640, 640), (485, 0, 1125, 640),
                            cons1=[np.array([[0, 0], [0, 569], [50, 569], [50, 0]])],
                            cons2=[np.array([[0, 0], [0, 639], [50, 639], [50, 0]])],
                            iou_threshold=0.5)

    fig, ax = plt.subplots()
    ax.imshow(image1, cmap='gray')
    for contour_index, con in enumerate(finals):
        con = np.squeeze(con)
        ax.plot(con[:, 0], con[:, 1], label=f"Contour {contour_index}")
    plt.title(f"Merged Contours, Contours: {len(finals)}")
    plt.show()
