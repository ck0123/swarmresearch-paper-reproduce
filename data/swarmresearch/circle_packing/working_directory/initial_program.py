# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles - hardcoded optimized solution."""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26

    centers = np.array([
        [0.889220987209202, 0.889220987209394],
        [0.915360499304285, 0.084639500695790],
        [0.240647598439205, 0.762958863654029],
        [0.705253940950931, 0.386923553409693],
        [0.904267670692995, 0.683258534973106],
        [0.314056978013262, 0.907407905048554],
        [0.500571630861010, 0.906072662722592],
        [0.728370148514024, 0.597634796387677],
        [0.686884190029652, 0.907608448428973],
        [0.103467233358018, 0.482595582210606],
        [0.502715553795886, 0.078860372915984],
        [0.404780267060306, 0.742049443460524],
        [0.896939479858421, 0.484600802651915],
        [0.705390511218109, 0.130221101065243],
        [0.501331924496604, 0.529963419753317],
        [0.105182560268725, 0.273952839623917],
        [0.297390396301023, 0.381665844452685],
        [0.596641216397275, 0.742417049501585],
        [0.096151334045768, 0.682080042932622],
        [0.760289472072460, 0.763673569383398],
        [0.504468239325592, 0.275342616771315],
        [0.111156179410508, 0.888843820589573],
        [0.893209855371477, 0.274783283350794],
        [0.297690474910829, 0.133258572770778],
        [0.084926262454926, 0.084926262454863],
        [0.273094285688398, 0.596042701908096],
    ])

    radii = np.array([
        0.110779012790741,
        0.084639500695769,
        0.069440193711300,
        0.112077088950281,
        0.095732329307078,
        0.092592094951494,
        0.093927337277515,
        0.099898350592786,
        0.092391551570968,
        0.103467233358001,
        0.078860372916102,
        0.096018975758303,
        0.103060520141654,
        0.130221101065367,
        0.137010430123808,
        0.105182560268820,
        0.115148880160095,
        0.095842325745576,
        0.096151334045792,
        0.069180676357282,
        0.117629688046583,
        0.111156179410542,
        0.106790144628586,
        0.133258572770807,
        0.084926262454915,
        0.100600367818770,
    ])

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    """
    n = centers.shape[0]
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
