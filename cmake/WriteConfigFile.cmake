MACRO(CudaFramework_WRITE_CONFIG_FILE CudaFramework_CONFIG_FILE CudaFramework_ROOT_DIR )

    # Get the version of the project CudaFramework
    execute_process(COMMAND "cmake" "-P" "cmake/GetGitRevisionDescriptionExtern.cmake" 
        WORKING_DIRECTORY "${CudaFramework_ROOT_DIR}"
        OUTPUT_VARIABLE CudaFramework_VERSION ERROR_VARIABLE Error
    )

    if(Error)
        message(FATAL_ERROR "Error in getting version of CudaFramework ${Error}" FATAL)
    endif()


    string(REGEX REPLACE "^.*v([0-9]+)\\..*" "\\1" CudaFramework_VERSION_MAJOR "${CudaFramework_VERSION}")
    string(REGEX REPLACE "^.*v[0-9]+\\.([0-9]+).*" "\\1" CudaFramework_VERSION_MINOR "${CudaFramework_VERSION}")
    string(REGEX REPLACE "^.*v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" CudaFramework_VERSION_PATCH "${CudaFramework_VERSION}")
    string(REGEX REPLACE "^.*v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" CudaFramework_VERSION_SHA1 "${CudaFramework_VERSION}")
    set(CudaFramework_VERSION_STRING "${CudaFramework_VERSION_MAJOR}.${CudaFramework_VERSION_MINOR}.${CudaFramework_VERSION_PATCH}")
    MESSAGE(STATUS "CudaFramework Version: ${CudaFramework_VERSION_STRING} extracted from git tags!")

    configure_file(
      ${CudaFramework_ROOT_DIR}/include/CudaFramework/General/ConfigureFile.hpp.in.cmake
      ${CudaFramework_CONFIG_FILE}
    )

ENDMACRO()

