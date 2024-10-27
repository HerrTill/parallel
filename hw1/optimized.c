#include <microtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float* Matrix;

// Create and initialize the matrix
Matrix createMatrix(int rows, int cols) {
    Matrix M = (Matrix)malloc(rows * cols * sizeof(M[0]));
    if (!M) {
        fprintf(stderr, "Matrix allocation failed in file %s, line %d\n", __FILE__, __LINE__);
    }
    return M;
}

void freeMatrix(Matrix M) {
    if (M) free(M);
}

// Initialize matrix with values A[i][j] = 1 / (i + j + 2)
void initMatrix(Matrix A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = 1.0f / (i + j + 2);
        }
    }
}

// Optimized matrix-vector multiplication
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
    // Initialize result vector C to 0
    memset(C, 0, rows * sizeof(C[0]));

    // Reordering loops for better memory access patterns
    // Loop unrolling and cache-friendly access pattern
    #pragma omp simd
    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < cols; k++) {
            C[i] += A[i * cols + k] * B[k];
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "USAGE: %s rows cols\n", argv[0]);
        exit(1);
    }

    int n = atoi(argv[1]);  // Number of rows
    int m = atoi(argv[2]);  // Number of columns
    int p = 1;              // Vector size (always 1 in this case)

    Matrix A = createMatrix(n, m);  // n x m matrix
    Matrix B = createMatrix(m, p);  // m x 1 vector
    Matrix C = createMatrix(n, p);  // n x 1 result vector

    initMatrix(A, n, m);
    initMatrix(B, m, p);
    memset(C, 0, n * p * sizeof(C[0]));

    // Measure the time for the matrix-vector multiplication
    double time1 = microtime();
    matVecMult(A, B, C, n, m);
    double time2 = microtime();

    double t = time2 - time1;

    // Print performance results
    printf("\nTime = %g us\n", t);
    printf("Timer Resolution = %g us\n", getMicrotimeResolution());
    printf("Performance = %g Gflop/s\n", 2.0 * n * m * 1e-3 / t);
    printf("C[N/2] = %g\n\n", (double)C[n / 2]);

    // Clean up memory
    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);

    return 0;
}
