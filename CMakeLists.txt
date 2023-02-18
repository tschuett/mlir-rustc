cmake_minimum_required(VERSION 3.25.0)

project(rust_compiler)

message(STATUS "Using LLVM in: ${LLVM_DIR}")

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")


find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using Cmake MLIR in: ${MLIR_CMAKE_DIR}")
message(STATUS "Using Cmake LLVM in: ${LLVM_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(LLVM_TABLEGEN_EXE "${LLVM_DIR}/../../../bin/llvm-tblgen")

message(STATUS "Using tablegen in: ${LLVM_TABLEGEN_EXE}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "cmake")

set(CMAKE_COMPILE_WARNING_AS_ERROR True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

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

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT output)
    if (ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "We are in release - Successfully enabled IPO")
    else ()
        message(WARNING "IPO NOT SUPPORTED - Skipping reason: ${output}")
    endif ()
endif ()

#find_package(GTest REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTest REQUIRED gtest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(code)
add_subdirectory(tools)
add_subdirectory(tests)

