// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================
#ifndef CudaFramework_General_CommonMacros_hpp
#define CudaFramework_General_CommonMacros_hpp


#define MACRO_MIN(x, y) (((x) <= (y)) ? (x) : (y))
#define MACRO_MAX(x, y) (((x) >= (y)) ? (x) : (y))
#define MACRO_MAX0(x) (((x) >= 0) ? (x) : 0)
#define MACRO_ABS(x) (((x) >= 0) ? (x) : (-x))

#define MACRO_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MACRO_DIV_ROUND(x, y) (((x) + (y) / 2) / (y))
#define MACRO_ROUND_UP(x, y) ((y) * MACRO_DIV_UP(x, y))
#define MACRO_SHIFT_DIV_UP(x, y) (((x) + ((1<< (y)) - 1))>> y)
#define MACRO_ROUND_UP_POW2(x, y) (((x) + (y) - 1) & ~((y) - 1))
#define MACRO_ROUND_DOWN_POW2(x, y) ((x) & ~((y) - 1))
#define MACRO_IS_POW_2(x) (0 == ((x) & ((x) - 1)))

#endif // CommonMacros_hpp
