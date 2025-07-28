import numpy as np


class CellularNoise:
    """
    A class to generate 2D Cellular (Worley) noise.

    Noise values are procedurally generated based on the distance to the nearest feature point in a 2D grid.
    It uses
    an adaptive traversal strategy to prune the search space when finding the closest point.
    Each cell in the grid
    contains one pseudo-randomly placed feature point,
    determined by a hashing scheme based on the cell's coordinates
    and a seed.
    """
    MCG_CONSTANT = 48271  # Constants for MCG hashing function
    MCG_MODULUS = 2147483647
    PRIME_1, PRIME_2 = 47242787, 60048899  # Large primes

    # Optimal traverse orders based on point location in cell
    TRAVERSE_ORDERS = [[1, 3, 11, 4, 12, 6, 15, 16, 0, 8, 9, 2, 10, 14, 5, 13, 17, 7, 18, 19],
                       [1, 4, 12, 3, 11, 6, 15, 16, 0, 8, 9, 5, 13, 17, 2, 10, 14, 7, 18, 19],
                       [6, 3, 15, 4, 16, 1, 11, 12, 7, 18, 19, 2, 10, 14, 5, 13, 17, 0, 8, 9],
                       [6, 4, 16, 3, 15, 1, 11, 12, 7, 18, 19, 5, 13, 17, 2, 10, 14, 0, 8, 9],
                       [3, 1, 11, 6, 15, 4, 12, 16, 2, 10, 14, 0, 8, 9, 7, 18, 19, 5, 13, 17],
                       [3, 6, 15, 1, 11, 4, 12, 16, 2, 10, 14, 7, 18, 19, 0, 8, 9, 5, 13, 17],
                       [4, 1, 12, 6, 16, 3, 11, 15, 5, 13, 17, 0, 8, 9, 7, 18, 19, 2, 10, 14],
                       [4, 6, 16, 1, 12, 3, 11, 15, 5, 13, 17, 7, 18, 19, 0, 8, 9, 2, 10, 14]]

    # Relative neighbors
    NEIGHBORS = [[-2, 0], [-1, 0], [0, -2], [0, -1], [0, 1], [0, 2], [1, 0], [2, 0], [-2, -1], [-2, 1],
                 [-1, -2], [-1, -1], [-1, 1], [-1, 2], [1, -2], [1, -1], [1, 1], [1, 2], [2, -1], [2, 1]]

    def __init__(self, seed: int | None = None):
        """
        Initializes the CellularNoise generator.

        Parameters
        ----------
        seed : int, optional
            An optional integer seed to for deterministic noise generation.
        """
        if seed is None:
            seed = np.random.randint(self.MCG_MODULUS)
        self.seed = np.int32(seed)
        self.mcg_seed = self._generate_seed(self.seed)

    def sample_noise(self, x: float, y: float) -> float:
        """
        Samples the noise value at the given location.

        Parameters
        ----------
        x : float
            The x-coordinate to sample.
        y : float
            The y-coordinate to sample.

        Returns
        -------
        float: The noise value at the specified coordinate.
        """
        i, j = self._square(x, y)       # Current cell
        x_rel, y_rel = x - i, y - j     # Point position in cell

        order = self._traversal_order(x_rel, y_rel)     # Optimal trav. order based on point's position in cell
        lower_bounds = self._lower_bounds(x_rel, y_rel)     # Lower bounds for pruning

        lowest_dist = self._evaluate_distance(x, y, i, j, np.inf)   # Start with distance to feature pt. of current cell

        for neighbor in self.TRAVERSE_ORDERS[order]:    # Traverse neighbors
            bound = lower_bounds[neighbor] if (neighbor < 8) else 0.0   # For adjacent neighbors lower bound is 0

            if lowest_dist > bound:
                neighbor_x, neighbor_y = self.NEIGHBORS[neighbor]   # Neighbors relative position to current cell
                neighbor_x += i
                neighbor_y += j
                # Evaluate distance to neighbor
                lowest_dist = self._evaluate_distance(x, y, neighbor_x, neighbor_y, lowest_dist)
            else:   # Skip neighbors whose lower bounds are larger than the lowest found distance
                break

        return lowest_dist

    @staticmethod
    def _square(x: float, y: float) -> tuple[int, int]:
        """
        Determines the starting coordinates of the grid cell containing a point.

        Parameters
        ----------
        x : float
            The x-coordinate of the point.
        y : float
            The y-coordinate of the point.

        Returns
        -------
        Tuple[int, int]: A tuple (i, j) containing the integer coordinates of the cell's
            bottom-left corner.
        """
        return np.floor(x).astype(int), np.floor(y).astype(int)

    @staticmethod
    def _traversal_order(x_rel: float, y_rel: float) -> int:
        """
        Determines the traversal order index for a point based on its relative position (x_rel, y_rel) within a cell.
        The function computes which of the eight possible pizza slices the point belongs to.
        It determines which axis (X or Y) is closest to the edge and then checks the relative position
        on the other axis to further refine the slice.

        The order is determined by:
        1. Identifying whether the X or Y coordinate is closer to an edge (indexes 0-1 or 2-3), and selecting
           the closest axis.
        2. If the point is closer to the X-axis edge, we know it's in the right or left two slices. We then determine
           whether it's in the left or right half of the slice depending on whether x_rel or 1-x_rel is argmin.
        3. It then checks whether it is the top or bottom of these two remaining slices by checking y_rel > 0.5.

        The function returns an index (0-7) corresponding to one of the eight pizza slices, which defines
        the optimal neighborhood visit order.

        Parameters
        ----------
        x_rel : float
            xi position relative to the beginning of the cell (0 to 1).
        y_rel : float
            yi position relative to the beginning of the cell (0 to 1).

        Returns
        -------
        int: An index from 0 to 7 representing the slice index based on the relative position.
        """
        distances = np.array([x_rel, 1 - x_rel, y_rel, 1 - y_rel], dtype=np.float64)
        smallest_distance = np.argmin(distances)

        if smallest_distance < 2:  # point is closer to edge on X than to Y -> we know its in left or right two slices
            other_axis = 1 if (y_rel > 0.5) else 0  # top half or bottom half
        else:  # point is closer to edge on Y than to X -> we know its in top or bottom two slices
            other_axis = 1 if (x_rel > 0.5) else 0  # left or right

        return smallest_distance * 2 + other_axis

    @staticmethod
    def _lower_bounds(x_rel: float, y_rel: float) -> list[float]:
        """
        Gets the lower bound distances to each of the eight axis-aligned neighbors for traversal pruning.
        If the distance to a neighbor is higher than the current closest, the whole row/column can be discarded.

        Parameters
        ----------
        x_rel : float
            The x position relative to the cell's origin (range 0 to 1).
        y_rel : float
            The y position relative to the cell's origin (range 0 to 1).

        Returns
        -------
        list[float]: A list of eight lower-bound distances to the axis-aligned neighbors.
        """
        return [1.0+x_rel, x_rel, 1.0+y_rel, y_rel, 1.0-y_rel, 2.0-y_rel, 1.0-x_rel, 2.0-x_rel]

    def _generate_seed(self, s1: int, s2: int | None = None):
        """
        Generates a pseudo-random 'seed', which is the starting number of the MCG.

        Parameters
        ----------
        s1 : int
            The first integer for hashing.
        s2 : int, optional
            The second integer for hashing for multiple inputs of the hashing function like coordinates.

        Returns
        -------
        int : The generated seed.
        """
        s1 = (s1 + self.seed) % self.MCG_MODULUS
        if s2 is None:
            self.mcg_seed = (s1 * self.PRIME_1) % self.MCG_MODULUS
            return self.mcg_seed
        else:
            s2 = (s2 + self.seed) % self.MCG_MODULUS
            self.mcg_seed = ((s1 * self.PRIME_1) % self.MCG_MODULUS) ^ ((s2 * self.PRIME_2) % self.MCG_MODULUS)
            return self.mcg_seed

    def _generate_float(self) -> float:
        """
        Generates the next pseudo-random float using an MCG.

        Returns
        -------
        float: A pseudo-random number in the range [0.0, 1.0).
        """
        self.mcg_seed = (self.mcg_seed * self.MCG_CONSTANT) % self.MCG_MODULUS
        return float(self.mcg_seed) / float(self.MCG_MODULUS)

    def _evaluate_distance(self, x: float, y: float, i: int, j: int, min_distance: float) -> float:
        """
        Evaluates the distance from a query point to the feature point in a specific cell.

        This function first generates the pseudo-random location of the feature
        point within the cell (i, j) and then calculates the Euclidean distance
        from the query point (x, y) to this feature point.

        It then returns the distance if its smaller than the given `min_distance`.

        Parameters
        ----------
        x : float
            The x-coordinate of the query point.
        y : float
            The y-coordinate of the query point.
        i : int
            The x-coordinate of the cell to evaluate.
        j : int
            The y-coordinate of the cell to evaluate.
        min_distance : float
            The smallest distance found so far in the search.

        Returns
        -------
        float: The new minimum distance, which is either the original `min_distance`
            or the distance to the feature point in this cell if it's smaller.

        """
        # Generate pseudo-random feature point
        self.mcg_seed = self._generate_seed(i, j)
        x_point_rel, y_point_rel = self._generate_float(), self._generate_float()

        # Get its coordinates
        point_x, point_y = i + x_point_rel, j + y_point_rel

        d_x, d_y = abs(x - point_x), abs(y - point_y)
        if (d_x < min_distance) or (d_y < min_distance):    # if distance along just one axis is higher -> can't be less
            # Return distance to this point if smaller than min_distance
            return min(min_distance, np.sqrt((d_x*d_x) + (d_y*d_y)))
        return min_distance


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    noise = CellularNoise()

    # Initialize empty image
    image = np.zeros((896, 896), dtype=np.float64)

    # Sample noise values for each pixel
    for yi in range(image.shape[0]):
        for xi in range(image.shape[1]):
            sampled_level = noise.sample_noise(x=xi * (10 / image.shape[0]), y=yi * (10 / image.shape[0]))
            image[yi, xi] = sampled_level

    # Normalize image
    image -= np.min(image)
    image = (image * (255 / np.max(image))).astype(np.uint8)
    # Plot image
    fig, arr = plt.subplots(nrows=1, ncols=1, layout='tight', figsize=(8, 8))
    arr.imshow(image, cmap='bone')
    plt.show()
