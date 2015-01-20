# this file is executed outside of CudaFramework to get the revision describtion
include(cmake/GetGitRevisionDescription.cmake)
git_describe(CudaFramework_VERSION "--tags" "--abbrev=0")
message(STATUS "${CudaFramework_VERSION}")