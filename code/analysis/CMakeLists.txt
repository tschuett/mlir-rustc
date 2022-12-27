add_library(analysis
           MemorySSA.cpp
           MemorySSAAnalysis.cpp
           MemorySSANode.cpp
           mssa.cpp
           )

target_include_directories(analysis PRIVATE  ../include)


target_link_libraries(analysis )
