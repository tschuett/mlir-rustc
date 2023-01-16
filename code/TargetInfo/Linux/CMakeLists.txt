add_library(Linux
           Linux.cpp
           AArch64.cpp
           AArch64TypeBuilder.cpp
           )

target_include_directories(Linux PRIVATE  ../../include)
