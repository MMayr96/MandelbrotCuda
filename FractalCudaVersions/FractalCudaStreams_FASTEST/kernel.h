#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "pfc_types.h"
#include<stdio.h>
#include<stdlib.h>
#include "constants.h"

//algorithm optimizations
#define BULB_CHECK

//color map in constant memory
__constant__ pfc::BGR_4_t color_map[MAX_ITERATIONS];

//register cpu generated color map as constant memory clolr map on gpu
cudaError_t register_color_map(const pfc::BGR_4_t* colors, const int n_colors);

//call mandelbrot kernel calculation
cudaError_t call_kernel_optimized(cudaStream_t stream, pfc::BGR_4_t *mandelbrot_dst, const dim3 big, const dim3 tib, const float4 boundary);