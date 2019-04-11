#include "kernel.h"

//calculates the iterations for a pixel
__device__ int mandelbrot(const float start_real, const float start_imaginary) {

	float zReal = start_real;
	float zImag = start_imaginary;

	// start bulb checking
	//https://en.wikipedia.org/wiki/Mandelbrot_set#Cardioid_/_bulb_checking
	float z_imag_squared = zImag * zImag;
	float z_real_minus_quarter = zReal - 0.25f;
	float z_real_plus_one = zReal + 1.0f;

	#ifdef BULB_CHECK
	float q = z_real_minus_quarter * z_real_minus_quarter + z_imag_squared;

	// The two equations determine that the point is within the cardioid, the last the period-2 bulb.
	float period_first_bulb = q * (q + z_real_minus_quarter);
	float period_second_bulb = z_real_plus_one * z_real_plus_one + z_imag_squared;

	if (period_first_bulb <= z_imag_squared * 0.25f || period_second_bulb <= 0.0625f) {
		return MAX_ITERATIONS-1;
	}
	#endif // BULB_CHECK

	//normal mandelbrot loop
	float r2, i2;
	#pragma unroll
	for (unsigned int counter{ 0 }; counter < MAX_ITERATIONS; counter++) {
		r2 = zReal * zReal;
		i2 = zImag * zImag;
		if (r2 + i2 > 4.0f) {
			return counter;
		}
		zImag = 2.0f * zReal * zImag + start_imaginary;
		zReal = r2 - i2 + start_real;
	}
	return MAX_ITERATIONS-1;
}

__global__ void mandelbrot_k_optimized(pfc::BGR_4_t *mandelbrot_dst, const float4 boundary) {

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	float range_real{ boundary.x - boundary.w }, range_imag{ boundary.z - boundary.y };
	float real = x * (range_real / IMAGE_WIDTH) + boundary.w;
	float imag = y * (range_imag / IMAGE_HEIGHT) + boundary.y;

	int iteration = mandelbrot(real, imag);
	//color pixel according to its needed iterations
	mandelbrot_dst[y * IMAGE_WIDTH + x] = color_map[iteration];
}

cudaError_t register_color_map(const pfc::BGR_4_t* colors, const int n_colors) {
	cudaMemcpyToSymbol(color_map, colors, sizeof(pfc::BGR_4_t)*(n_colors));
	return cudaGetLastError();
}
 
cudaError_t call_kernel_optimized(cudaStream_t stream, pfc::BGR_4_t *mandelbrot_dst, const dim3 big, const dim3 tib, const float4 boundary) {
	mandelbrot_k_optimized <<<big, tib,0,stream>>>(mandelbrot_dst, boundary);
	return cudaGetLastError();
}

    


