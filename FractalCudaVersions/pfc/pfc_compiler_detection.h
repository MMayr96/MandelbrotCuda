//           $Id: pfc_compiler_detection.h 37984 2018-10-27 15:50:30Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2018-WS/ILV/src/Snippets/bitmap-gsl/pfc_compiler_detection.h $
//     $Revision: 37984 $
//         $Date: 2018-10-27 17:50:30 +0200 (Sa., 27 Okt 2018) $
//       $Author: p20068 $
//
//       Creator: Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
// Creation Date:
//     Copyright: (c) 2018 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//
//       License: This document contains proprietary information belonging to
//                University of Applied Sciences Upper Austria, Campus
//                Hagenberg. It is distributed under the Boost Software License,
//                Version 1.0 (see http://www.boost.org/LICENSE_1_0.txt).
//
//    Annotation: This file is part of the code snippets handed out during one
//                of my HPC lessons held at the University of Applied Sciences
//                Upper Austria, Campus Hagenberg.
//
// see https://sourceforge.net/p/predef/wiki/Compilers
//     http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-identification-macro

#pragma once

#include <type_traits>

// -------------------------------------------------------------------------------------------------

#undef PFC_DETECTED_COMPILER_CL
#undef PFC_DETECTED_COMPILER_CLANG
#undef PFC_DETECTED_COMPILER_GCC
#undef PFC_DETECTED_COMPILER_ICC
#undef PFC_DETECTED_COMPILER_NONE
#undef PFC_DETECTED_COMPILER_NVCC
#undef PFC_DETECTED_COMPILER_TYPE

#if defined __clang__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_clang_t
   #define PFC_DETECTED_COMPILER_CLANG

#elif defined __CUDACC__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_nvcc_t
   #define PFC_DETECTED_COMPILER_NVCC

#elif defined __GNUC__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_gcc_t
   #define PFC_DETECTED_COMPILER_GCC

#elif defined __INTEL_COMPILER
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_icc_t
   #define PFC_DETECTED_COMPILER_ICC

#elif defined _MSC_VER
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_cl_t
   #define PFC_DETECTED_COMPILER_CL

#else
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_none_t
   #define PFC_DETECTED_COMPILER_NONE
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_KNOW_PRAGMA_WARNING_PUSH_POP

#if defined PFC_DETECTED_COMPILER_CL || defined PFC_DETECTED_COMPILER_NVCC
   #define PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#endif

// -------------------------------------------------------------------------------------------------

namespace pfc {

enum class compiler {
   none, cl, clang, gcc, icc, nvcc
};

using detected_compiler_none_t  = std::integral_constant <compiler, compiler::none>;
using detected_compiler_cl_t    = std::integral_constant <compiler, compiler::cl>;
using detected_compiler_clang_t = std::integral_constant <compiler, compiler::clang>;
using detected_compiler_gcc_t   = std::integral_constant <compiler, compiler::gcc>;
using detected_compiler_icc_t   = std::integral_constant <compiler, compiler::icc>;
using detected_compiler_nvcc_t  = std::integral_constant <compiler, compiler::nvcc>;

using detected_compiler_t = PFC_DETECTED_COMPILER_TYPE;

constexpr auto detected_compiler_v {detected_compiler_t::value};

constexpr auto detected_compiler () noexcept {
   return detected_compiler_v;
}

}   // namespace pfc
