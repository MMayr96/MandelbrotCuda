#include "kernel.h"
#include "pfc_base.h"
#include "pfc_bitmap_3.h"
#include "func_timer.h"
#include "thread.cpp"
#include <gsl/gsl>

//#define SAVE_IMAGES
#define LOG_TO_FILE

//check for error
void check(cudaError_t const error) {
	if (error != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << '\n'; //exit(1);
	}
}

//helper func to calc number of threads
int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

//initializes calculation bounds for pic 0 to max num pictures
void initialize_calculations_bounds(float4* bounds) {

	bounds[0].w = REAL_MIN_START;
	bounds[0].x = REAL_MAX_START;
	bounds[0].y = IMAG_MIN_START;
	bounds[0].z = IMAG_MAX_START;

	for (unsigned int i{ 1 }; i < NUM_PICTURES; i++) {
		bounds[i].w = TARGET_REAL + (bounds[i - 1].w - TARGET_REAL) * ZOOM_FACTOR;
		bounds[i].x = TARGET_REAL + (bounds[i - 1].x - TARGET_REAL) * ZOOM_FACTOR;
		bounds[i].y = TARGET_IMAG + (bounds[i - 1].y - TARGET_IMAG) * ZOOM_FACTOR;
		bounds[i].z = TARGET_IMAG + (bounds[i - 1].z - TARGET_IMAG) * ZOOM_FACTOR;
	}
}

// initialize color map
void initialize_color_map(pfc::BGR_4_t* colors) {

	for (unsigned int i{ 0 }; i < MAX_ITERATIONS; i++) {
		float t = static_cast<float>(i) / MAX_ITERATIONS;
		// Use smooth polynomials for r, g, b
		colors[i].red = static_cast<pfc::byte_t>(9 * (1 - t)*t*t*t * 255);
		colors[i].green = static_cast<pfc::byte_t>(15 * (1 - t)*(1 - t)*t*t * 255);
		colors[i].blue = static_cast<pfc::byte_t>(8.5*(1 - t)*(1 - t)*(1 - t)*t * 255);
	}
}

void mandelbrot_optimized() {

#ifdef LOG_TO_FILE
	freopen("output.txt", "w", stdout);
#endif

	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y); //tib
	dim3 grid(divup(IMAGE_WIDTH, BLOCKDIM_X), divup(IMAGE_HEIGHT, BLOCKDIM_Y)); //big

	//initialize color palette
	pfc::BGR_4_t colors[MAX_ITERATIONS];
	initialize_color_map(colors);
	//register color map on gpu as constant. register is only a wrapper function
	check(register_color_map(colors, MAX_ITERATIONS));

	//initialize boundaries
	float4 bounds[NUM_PICTURES];
	initialize_calculations_bounds(bounds);

	//calculate mermory needed for one mandelbrot frame
	std::size_t mandelbrot_size = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(pfc::BGR_4_t);

	//declare host/device pointer with respect to streams
	pfc::BGR_4_t *h_mandelbrot[NUM_STREAMS], *d_mandelbrot[NUM_STREAMS];
	pfc::bitmap bmp[NUM_STREAMS];
	
	//setup streams
	cudaStream_t streams[NUM_STREAMS];
	int stream_nr = 0, streams_busy = 0;
	bool stream_busy[NUM_STREAMS];

	for (int i = 0; i < NUM_STREAMS; i++) {
		//allocate memory on device
		check(cudaMalloc(&d_mandelbrot[i], mandelbrot_size));
		//create stream i
		check(cudaStreamCreate(&streams[i]));
		//allocate pinned host memory on host
		check(cudaMallocHost(&h_mandelbrot[i], mandelbrot_size));
		//create bitmap with pixel_span pointer
		bmp[i] = pfc::bitmap{ IMAGE_WIDTH, IMAGE_HEIGHT, gsl::span<pfc::BGR_4_t>(h_mandelbrot[i],IMAGE_WIDTH*IMAGE_HEIGHT) }; //v4
		stream_busy[i] = false;
	}

	std::chrono::time_point<std::chrono::steady_clock> timings_start[NUM_PICTURES];
	std::chrono::time_point<std::chrono::steady_clock> timings_end[NUM_PICTURES];
	auto time_start = std::chrono::high_resolution_clock::now();

	int zoom_idx = 0;
	//iterate over numpictures and split into streams
	while (zoom_idx < NUM_PICTURES || streams_busy) {

		if (stream_busy[stream_nr]) {
			stream_busy[stream_nr] = false; --streams_busy;
			check(cudaStreamSynchronize(streams[stream_nr]));

			timings_end[zoom_idx] = std::chrono::high_resolution_clock::now();

			//save the image
			#ifdef SAVE_IMAGES
			bmp[stream_nr].to_file(IMAGE_DIR + std::to_string(zoom_idx) + IMAGE_TYPE);
			#endif // SAVE_IMAGES
			if (zoom_idx + 1 >= NUM_PICTURES) {
				break;
			}
			zoom_idx++;
		}

		if (zoom_idx < NUM_PICTURES) {
			//frame timing start
			timings_start[zoom_idx] = std::chrono::high_resolution_clock::now();
			//call mandelbrot kernel
			check(call_kernel_optimized(streams[stream_nr],d_mandelbrot[stream_nr], grid, threads, bounds[zoom_idx]));
			//copy results async to host
			check(cudaMemcpyAsync(h_mandelbrot[stream_nr], d_mandelbrot[stream_nr], mandelbrot_size, cudaMemcpyDeviceToHost, streams[stream_nr]));
			stream_busy[stream_nr] = true; ++streams_busy;
		}
		//switch to new stream
		stream_nr = (stream_nr + 1) % NUM_STREAMS;
	}
	//end total time
	auto time_end = std::chrono::high_resolution_clock::now();

	//free up host and device memory
	for (int i = 0; i < NUM_STREAMS; i++) {
		check(cudaFree(d_mandelbrot[i]));
		check(cudaFreeHost(h_mandelbrot[i]));
	}
	//reset device
	check(cudaDeviceReset());

	//print timings (per frame and then total)
	std::cout << "PicIdx" << "," << "ms" << "," << "MiB/s" << std::endl;
	for (int i{ 0 }; i < NUM_PICTURES; i++) {
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timings_end[i] - timings_start[i]).count();
		const auto mebibytes_calculated = static_cast<long long>(NUM_PICTURES) * IMAGE_WIDTH * IMAGE_HEIGHT * 4.0;
		const auto mibs = mebibytes_calculated / duration * 1000 / 1024 / 1024;
		std::cout << i << "," << duration << "," << mibs << std::endl;
	}
	std::cout << "Total computation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << "ms" << std::endl;
}

int main() {

	int count{ -1 };
	cudaGetDeviceCount(&count);

	if (count <= 0) {
		std::cerr << "No Cuda Device!" << std::endl;
		exit(1);
	}

	cudaSetDevice(0);
	cudaDeviceProp prop;
	check(cudaGetDeviceProperties(&prop, 0));

	std::cout << "name: " << prop.name << "\ncc: " << prop.major << '.' << prop.minor << '\n';

	//call mandelbrot on gpu
	mandelbrot_optimized();

	return 0;
}