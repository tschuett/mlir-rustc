cmake_minimum_required(VERSION 3.24.0)

project(rust_compiler)

enable_testing()
include(CTest)



set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "cmake")

set(CMAKE_COMPILE_WARNING_AS_ERROR True)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
   set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -march=native")
endif()

CHECK_CXX_COMPILER_FLAG("-fproc-stat-report" COMPILER_SUPPORTS_PROC_STAT_REPORT)
if(COMPILER_SUPPORTS_PROC_STAT_REPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fproc-stat-report=abc")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)

#find_package(GTest REQUIRED)

add_subdirectory(code)
add_subdirectory(tools)

