#include <stdlib.h>
#include <stddef.h>
#include <math.h>

double square_dist(double *x, size_t len, double *v) {
	double sqd = 0.0;
	for (int i = 0; i < len; ++i) {
		double d = x[i] - v[i];
		sqd += d * d;
	}
	return sqd;
}

void calculate_inv_dists(double *coords, size_t n_rows, double *output,
size_t out_size) {
	int o = 0;
	for (int i = 0; i < n_rows && o < out_size; ++i) {
		for (int j = i + 1; j < n_rows && o < out_size; ++j) {
			double sq_dist = square_dist(&coords[i*3], 3, &coords[j*3]);
			output[o++] = 1.0 / sqrt(sq_dist);
		}
	}
}

double two_pass_rbf_var(double *vals, int size, double b) {
	double mean = 0.0;
	double var = 0.0;
	for (int i = 0; i < size; ++i) {
		mean += exp(-b * vals[i]);
	}
	mean /= size;
	for (int i = 0; i < size; ++i) {
		double v = exp(-b * vals[i]) - mean;
		var += v * v;
	}
	return var / size;
}

double one_pass_rbf_var(double *vals, int size, double b) {
	double mean = 0.0;
	double sq_mean = 0.0;
	for (int i = 0; i < size; ++i) {
		double v = exp(-b * vals[i]);
		mean += v;
		sq_mean += v * v;
	}
	mean /= size;
	sq_mean /= size;
	double var = sq_mean - mean * mean;
	return var;
}

double rbf_maximise_variance(double *features, size_t n_rows, size_t n_cols,
int u, int l) {
	int n_dists = (n_rows * (n_rows - 1)) / 2;
	int n_bands = (int) ((u - l) / 0.01) + 1;
	double *dists = NULL;
	dists = malloc(n_dists * sizeof(double));
	if (!dists) goto error;
	int p = 0;
	for (int i = 1; i < n_rows; ++i) {
		for (int j = 0; j < i; ++j) {
			dists[p++] = square_dist(&features[i*n_cols], n_cols,
				&features[j*n_cols]);
		}
	}
	double b;
	int lo = 1, hi = n_bands - 2;
	while (lo <= hi) {
		int m = lo + (hi - lo) / 2;
		double var_m, var_left, var_right;
		b = pow(10, l + 0.01 * m);
		var_m = one_pass_rbf_var(dists, n_dists, b);
		b = pow(10, l + 0.01 * (m - 1));
		var_left = one_pass_rbf_var(dists, n_dists, b);
		if (var_left > var_m) {
			hi = m - 1;
			continue;
		}
		b = pow(10, l + 0.01 * (m + 1));
		var_right = one_pass_rbf_var(dists, n_dists, b);
		if (var_right > var_m) {
			lo = m + 1;
		}
		else {
			free(dists);
			b = pow(10, l + 0.01 * m);	
			return 1.0 / sqrt(2.0 * b);
		}
	}
	error:
	free(dists);
	return -1.0;
}
