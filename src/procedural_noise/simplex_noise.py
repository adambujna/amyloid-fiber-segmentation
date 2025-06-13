import numpy as np


class SimplexNoise2D:
    def __init__(self, octave_count=1, scale=70.0, seed=None):
        self.octave_count = octave_count
        self.scale = scale
        self.seed = seed
        # Combining multiple octaves for fine-grained noise
        self.octaves = [self.SimplexOctave2D(scale=self.scale, seed=self.seed) for _ in range(self.octave_count)]
        self.frequencies = [2**i for i in range(self.octave_count)]
        self.amplitudes = [0.5**i for i in range(self.octave_count)]

    def sample_noise(self, x, y):
        """
        Samples the noise at specific coordinates

        Parameters
        ----------
        x : Number
            xi coordinate
        y : Number
            yi coordinate
        """
        combined_noise = [self.octaves[i].sample_noise(x=(x * self.frequencies[i]),
                                                       y=(y * self.frequencies[i])
                                                       ) * self.amplitudes[i]
                          for i in range(self.octave_count)]
        return np.sum(combined_noise)

    class SimplexOctave2D:
        F = 0.5 * (np.sqrt(3) - 1.0)  # Skew factor
        G = (3.0 - np.sqrt(3)) / 6.0  # Unskew constant
        
        def __init__(self, scale=70.0, seed=None):
            self.scale = scale
            self._rng = np.random.default_rng(seed=seed)

            # Permutation table (for pseudo-random deterministic generation)
            self.perm = np.arange(start=0, stop=256, dtype=np.uint8)
            self._rng.shuffle(self.perm)
            self.perm = np.tile(self.perm, reps=2)  # No need for wraparound when adding xi % 255 + perm[yi]
            self._perm_mod4 = self.perm % 4  # Avoid recalculating mod4 to get one of 4 gradients for each corner

            # Possible gradients in 2D
            self._gradients = [[1, 1], [-1, 1], [1, -1], [-1, -1]]

        def sample_noise(self, x, y):
            i, j = self._get_cell(x, y)     # Determine cell point (xi, yi) is in
            xc, yc = self._unskew(i, j)     # Get xi,yi coordinates of cell
            x0, y0 = x - xc, y - yc   # Distances from cell start

            # Determine if we are in the top or bottom simplex of cell
            if x0 > y0:
                i1, j1 = 1, 0       # Lower triangle order (0,0)->(1,0)->(1,1)
            else:
                i1, j1 = 0, 1       # Upper triangle order (0,0)->(0,1)->(1,1)

            # A step of (1,0) in skewed coordinates means a step of (1-G,-G) in unskewed,
            # a step of (0,1) means a step of (-G,1-G)
            x1, y1 = x0 - i1 + self.G, y0 - j1 + self.G
            x2, y2 = x0 - 1.0 + (2*self.G), y0 - 1.0 + (2*self.G)

            # Get hashed gradient indexes of simplex corners
            ii, jj = i & 255, j & 255     # Bit-wise AND to fit 256-size permutation table
            gi0 = self._perm_mod4[ii + int(self.perm[jj])]
            gi1 = self._perm_mod4[ii + i1 + int(self.perm[jj + j1])]
            gi2 = self._perm_mod4[ii + 1 + int(self.perm[jj + 1])]

            contributions = self._get_contributions(points=[[x0, y0], [x1, y1], [x2, y2]],
                                                    gradient_indexes=[gi0, gi1, gi2])

            return self.scale * np.sum(contributions)

        def _get_cell(self, x_sample, y_sample):
            """Gets the cell that the point is in."""
            factor = (x_sample + y_sample) * self.F
            x1 = x_sample + factor
            y1 = y_sample + factor
            return int(np.floor(x1)), int(np.floor(y1))

        def _unskew(self, x1, y1):
            factor = (x1 + y1) * self.G
            return (x1 - factor), y1 - factor

        def _get_contributions(self, points, gradient_indexes):
            """
            Gets contribution of each corner (multiplication of gradient extrapolation and attenuation function).
            Attenuation function is f(xi) = 0.5 - r_x^2 - r_y^2 (it ensures only the three points around sample matter).
            """
            contributions = []

            for p, gi in zip(points, gradient_indexes):
                x, y = p[0], p[1]
                t = 0.5 - x*x - y*y

                if t < 0:
                    n = 0.0
                else:
                    t *= t
                    n = t*t * np.dot(a=self._gradients[gi], b=[x, y])
                contributions.append(n)

            return contributions


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    noise = SimplexNoise2D(octave_count=8, scale=40.0)
    image = np.zeros((896, 896), dtype=np.uint8)
    for yi in range(image.shape[0]):
        for xi in range(image.shape[1]):
            sampled_level = noise.sample_noise(x=xi * (1.5 / image.shape[1]), y=yi * (1.5 / image.shape[0]))
            image[yi, xi] = ((sampled_level + 1) / 2) * 255

    image_inverted = ~image
    fig, arr = plt.subplots(nrows=1, ncols=2, layout='tight', figsize=(16, 8))
    arr[0].imshow(image, cmap='magma')
    arr[1].imshow(image_inverted, cmap='magma')
    plt.show()
