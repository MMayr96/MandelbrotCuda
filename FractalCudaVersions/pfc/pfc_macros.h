//           $Id: pfc_macros.h 37984 2018-10-27 15:50:30Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2018-WS/ILV/src/Snippets/bitmap-gsl/pfc_macros.h $
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

#pragma once

#include "./pfc_compiler_detection.h"

// -------------------------------------------------------------------------------------------------

#undef  PFC_MACRO_EXPAND
#define PFC_MACRO_EXPAND(x) \
   x

#undef  PFC_STATIC_ASSERT
#define PFC_STATIC_ASSERT(c) \
   static_assert ((c), PFC_STRINGIZE (c))

#undef  PFC_STRINGIZE
#define PFC_STRINGIZE(x) \
   #x

// -------------------------------------------------------------------------------------------------

#undef CATTR_DEVICE
#undef CATTR_GPU_ENABLED
#undef CATTR_GPU_ENABLED_INL
#undef CATTR_HOST
#undef CATTR_INLINE
#undef CATTR_RESTRICT

#if defined PFC_DETECTED_COMPILER_NVCC
   #define CATTR_DEVICE   __device__
   #define CATTR_HOST     __host__
   #define CATTR_INLINE   __forceinline__
   #define CATTR_RESTRICT __restrict__
#else
   #define CATTR_DEVICE
   #define CATTR_HOST
   #define CATTR_INLINE   inline
   #define CATTR_RESTRICT __restrict
#endif

#define CATTR_GPU_ENABLED        CATTR_HOST CATTR_DEVICE
#define CATTR_GPU_ENABLED_INLINE CATTR_HOST CATTR_DEVICE CATTR_INLINE
