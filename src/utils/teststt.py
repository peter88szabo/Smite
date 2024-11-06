import numpy as np

# Original function
def cenmass_original(q, w):
    qcm = np.zeros(3)
    
    # Determine the center of mass
    for i in range(len(w)):
        qcm[0] += q[3 * i]     * w[i]
        qcm[1] += q[3 * i + 1] * w[i]
        qcm[2] += q[3 * i + 2] * w[i]
    
    qcm /= sum(w)  # Normalize by total weight
    return qcm

# Simplified function
def cenmass_simplified(q, w):
    # Reshape q to separate x, y, z coordinates for each particle, then compute weighted average
    q = np.reshape(q, (-1, 3))
    qcm = np.average(q, axis=0, weights=w)
    return qcm

# Test function to verify both functions produce the same output
def test_cenmass():
    # Test data: Example coordinates (flattened 3D positions for each particle)
    q = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]  # Three particles
    w = [1.0, 2.0, 3.0]  # Weights for each particle

    # Calculate center of mass using both functions
    result_original = cenmass_original(q, w)
    result_simplified = cenmass_simplified(q, w)

    # Print results for comparison
    print("Center of Mass (Original):", result_original)
    print("Center of Mass (Simplified):", result_simplified)

    # Check if they are equal
    if np.allclose(result_original, result_simplified):
        print("Test passed: Both functions produce the same result.")
    else:
        print("Test failed: Results differ between the two functions.")

# Run the test
if __name__ == '__main__':
    test_cenmass()

