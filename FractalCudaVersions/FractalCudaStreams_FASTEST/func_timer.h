#pragma once
//       $Id: chrono.cpp 38132 2018-11-21 07:41:21Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/DSE/HPC3/2018-WS/ILV/src/Snippets/chrono.cpp $
// $Revision: 38132 $
//     $Date: 2018-11-21 08:41:21 +0100 (Mi., 21 Nov 2018) $
//   $Author: p20068 $
//
//   Creator: Peter Kulczycki
//  Creation: November 14, 2018
// Copyright: (c) 2018 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License, Version 1.0
//            (see http://www.boost.org/LICENSE_1_0.txt).
#include <chrono>

template <typename clock_t = std::chrono::steady_clock, typename size_t, typename fun_t, typename ...args_t>
inline auto timed_run(size_t const n, fun_t && fun, args_t && ...args) noexcept (std::is_nothrow_invocable_v <fun_t, args_t...>) {
	static_assert (clock_t::is_steady);
	static_assert (std::is_integral_v <size_t>);

	typename clock_t::duration elapsed{};

	if (0 < n) {
		auto const start{ clock_t::now() };
		for (int i{ 0 }; i < n; ++i) {
			std::invoke(std::forward <fun_t>(fun), std::forward <args_t>(args)...);
		}
		elapsed = (clock_t::now() - start) / n;
	}
	return elapsed;
}
