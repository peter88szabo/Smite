from prettytable import PrettyTable
import numpy as np

from prettytable import PrettyTable
import numpy as np

def print_matrix_pretty(matrix,title,float_format="{:.6e}"):

    # Convert 1D array to 2D array with one row
    if isinstance(matrix, np.ndarray) and matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=0)


    if isinstance(matrix, np.ndarray):
        if matrix.size == 0:
            print("The matrix is empty.")
            return
        rows, cols = matrix.shape
    elif isinstance(matrix, list):
        if not matrix:
            print("The matrix is empty.")
            return
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0

    # Create a PrettyTable object with the size in the header
    table = PrettyTable(header=False, border=False, hrules=False, vrules=False)

    for row in matrix:
        #table.add_row([f"{num:.6e}" for num in row])
        table.add_row([float_format.format(num) for num in row])

    print(title, f"(Size: {rows} x {cols})")
    print(table, "\n")

if __name__ == '__main__':

    matrix1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]


    matrix2 = np.array([
    [9.94099885e-01, -2.32889098e-01, 1.65239924e-08, 1.00235369e-01, -3.85344179e-16, -1.35631603e-01, 5.67754109e-08],
    [2.67799061e-02, 8.31788029e-01, -9.01654840e-08, -5.23423147e-01, 2.05378182e-15, 9.08581109e-01, -4.29532705e-07],
    [3.46629828e-03, 1.03349380e-01, -3.46565855e-01, 6.48259146e-01, -3.08201907e-15, 5.83295637e-01, 5.82525064e-01],
    [1.42075891e-19, 6.84153921e-17, -6.56688967e-16, 3.76145571e-15, 1.00000000e+00, -7.43540932e-17, -6.04603819e-17],
    [2.45105477e-03, 7.30794063e-02, 4.90116058e-01, 4.58390415e-01, -1.33250799e-15, 4.12453687e-01, -8.23811715e-01],
    [-6.08393404e-03, 1.60224007e-01, 4.41542345e-01, 2.69085782e-01, -9.52194530e-16, -8.07337325e-01, 8.42614908e-01],
    [-6.08393255e-03, 1.60223964e-01, -4.41542351e-01, 2.69085844e-01, -1.28842352e-15, -8.07337848e-01, -8.42614234e-01]
    ])


    matrix3 = np.array([[-20.23339442,  -1.26583942,  -0.62936507,  -0.44172496,  -0.38767176, 0.60308241,   0.76613482]])
    matrix4 = np.array([-20.23339442,  -1.26583942,  -0.62936507,  -0.44172496,  -0.38767176, 0.60308241,   0.76613482])

    print_matrix_pretty(matrix1, "matrix_1:", "{:.6e}")
    print_matrix_pretty(matrix2, "matrix_2:", "{:12.2e}")
    print_matrix_pretty(matrix3, "matrix_3:", "{:12.0f}")
    print_matrix_pretty(matrix4, "matrix_4:")

