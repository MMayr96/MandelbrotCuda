#pragma once

//fractal properties
constexpr int IMAGE_HEIGHT{ 4608 };
constexpr int IMAGE_WIDTH{ 8192 };
constexpr int MAX_ITERATIONS{ 128 };

//launch properties
constexpr unsigned int BLOCKDIM_X{ 32 };
constexpr unsigned int BLOCKDIM_Y{ 4 };

//images properties
constexpr float ZOOM_FACTOR{ 0.95f };
constexpr int NUM_PICTURES{ 200 };
constexpr const char* IMAGE_DIR{ "images/" };
constexpr const char* IMAGE_TYPE{ ".bmp" };

//target
constexpr float TARGET_REAL{ -0.745289981f };
constexpr float TARGET_IMAG{ 0.113075003f };

//start boundary
constexpr float REAL_MIN_START{ -2.74529004f };
constexpr float REAL_MAX_START{ 1.25470996f };
constexpr float IMAG_MIN_START{ -1.01192498f };
constexpr float IMAG_MAX_START{ 1.23807502f };

//streams
constexpr int NUM_STREAMS{ 5 };
