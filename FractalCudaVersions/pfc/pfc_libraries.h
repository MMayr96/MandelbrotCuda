//           $Id: pfc_libraries.h 37984 2018-10-27 15:50:30Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2018-WS/ILV/src/Snippets/bitmap-gsl/pfc_libraries.h $
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

#include "./pfc_macros.h"

// -------------------------------------------------------------------------------------------------

#if defined PFC_DETECTED_COMPILER_NVCC
   #define PFC_DO_NOT_USE_BOOST_UNITS
   #define PFC_DO_NOT_USE_GSL
   #define PFC_DO_NOT_USE_VLD
   #define PFC_DO_NOT_USE_WINDOWS
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_HAVE_VLD
#undef PFC_VLD_INCLUDED

#if __has_include (<vld.h>) && !defined PFC_DO_NOT_USE_VLD   // Visual Leak Detector (https://kinddragon.github.io/vld)
   #include <vld.h>

   #define PFC_HAVE_VLD
   #define PFC_VLD_INCLUDED

   #pragma message ("PFC: using 'Visual Leak Detector'")
#else
   #pragma message ("PFC: not using 'Visual Leak Detector'")
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_HAVE_GSL
#undef PFC_GSL_INCLUDED

#if __has_include (<gsl/gsl>) && !defined PFC_DO_NOT_USE_GSL   // Guideline Support Library (https://github.com/Microsoft/GSL)
   #include <gsl/gsl>

   #define PFC_HAVE_GSL
   #define PFC_GSL_INCLUDED

   #pragma message ("PFC: using 'Guideline Support Library'")
#else
   #pragma message ("PFC: not using 'Guideline Support Library'")
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_HAVE_BOOST_UNITS
#undef PFC_BOOST_UNITS_INCLUDED

#if __has_include (<boost/units/io.hpp>)
#if __has_include (<boost/units/systems/si/length.hpp>)
#if __has_include (<boost/units/systems/si/prefixes.hpp>) && !defined PFC_DO_NOT_USE_BOOST_UNITS
   #include <boost/units/io.hpp>                    // http://www.boost.org
   #include <boost/units/systems/si/length.hpp>     // https://sourceforge.net/projects/boost/files/boost-binaries
   #include <boost/units/systems/si/prefixes.hpp>   //

   #define PFC_HAVE_BOOST_UNITS
   #define PFC_BOOST_UNITS_INCLUDED

   #pragma message ("PFC: using 'Boost.Units'")
#else
   #pragma message ("PFC: not using 'Boost.Units'")
#endif
#endif
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_HAVE_WINDOWS
#undef PFC_WINDOWS_INCLUDED

#if __has_include (<windows.h>) && !defined PFC_DO_NOT_USE_WINDOWS
   #undef  NOMINMAX
   #define NOMINMAX

   #undef  STRICT
   #define STRICT

   #undef  VC_EXTRALEAN
   #define VC_EXTRALEAN

   #undef  WIN32_LEAN_AND_MEAN
   #define WIN32_LEAN_AND_MEAN

   #include <windows.h>

   #define PFC_HAVE_WINDOWS
   #define PFC_WINDOWS_INCLUDED

   #pragma message ("PFC: using 'windows.h'")
#else
   #pragma message ("PFC: not using 'windows.h'")
#endif
