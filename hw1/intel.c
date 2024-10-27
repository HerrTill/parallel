#include <immintrin.h>  // For AVX intrinsics
#include <string.h>     // For memset
#include <microtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float* Matrix;

// Optimized matrix-vector multiplication using AVX
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
    memset(C, 0, rows * sizeof(C[0]));  // Initialize C to 0

    // Process 8 floats at a time (AVX can handle 8 floats using 256-bit registers)
    for (int i = 0; i < rows; i++) {
        __m256 sum = _mm256_setzero_ps();  // Initialize sum to 0

        for (int k = 0; k < cols; k += 8) {  // Process 8 columns at a time
            // Load 8 elements from matrix A and vector B
            __m256 a_vals = _mm256_loadu_ps(&A[i * cols + k]);
            __m256 b_vals = _mm256_loadu_ps(&B[k]);

            // Perform element-wise multiplication
            __m256 prod = _mm256_mul_ps(a_vals, b_vals);

            // Accumulate the results
            sum = _mm256_add_ps(sum, prod);
        }

        // Horizontal sum of the 8 floats in the SIMD register
        // sum = (sum[0] + sum[1] + ... + sum[7])
        float result[8];
        _mm256_storeu_ps(result, sum);  // Store the 8-element vector back to memory
        C[i] = result[0] + result[1] + result[2] + result[3] +
               result[4] + result[5] + result[6] + result[7];  // Accumulate sum
    }
}

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

