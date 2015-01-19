

PROJECT(TestCudaModern)
message(STATUS "==============================Project: ${PROJECT_NAME} ==================================")
include(PrintListMacro)

# Set  Sources and Includes
SET(SOURCE_FILES
    ${GeneralCommonFiles_SRC}
    ${COMMON_FILES_SRC}
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

SET(INCLUDE_FILES
    ${GeneralCommonFiles_SRC}
    ${COMMON_FILES_INC}
)


SET(INCLUDE_DIRS 
${EIGEN_INCLUDE_DIR}
${COMMON_INCLUDE_DIR}
)
include_directories(${INCLUDE_DIRS})
PRINTLIST("Include Directories are:" "${INCLUDE_DIRS}")

#link directories
set(LINK_DIRS)
link_directories(${LINK_DIRS})
PRINTLIST("Link directories are:" "${LINK_DIRS}")
set(LINKED_LIBS )
PRINTLIST("Linked libraries are:" "${LINKED_LIBS}")

set_source_files_properties( INCLUDE_FILES PROPERTIES HEADER_FILE_ONLY TRUE)

#The rule to build the executable program
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES} ${INCLUDE_FILES} )
target_link_libraries(${PROJECT_NAME}  ${LINKED_LIBS})
