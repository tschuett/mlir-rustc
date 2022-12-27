add_library(analysis
           MemorySSA.cpp
           AliasAnalysis.cpp
           )

target_include_directories(analysis PRIVATE  ../include)


target_link_libraries(analysis )
