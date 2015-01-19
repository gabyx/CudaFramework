// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================
#ifndef CudaFramework_General_PlatformDefines_hpp
#define CudaFramework_General_PlatformDefines_hpp



#if _WIN32 || _WIN64
    #define HOLD_SYSTEM { system("pause"); }
#else
    #define HOLD_SYSTEM { \
    printf("Press 'Enter' to exit the program ..."); \
    while (getchar() != '\n'); \
    printf("\n\n");}
#endif

#endif
