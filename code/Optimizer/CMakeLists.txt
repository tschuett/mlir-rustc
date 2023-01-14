add_library(optimizer
           Attributer.cpp
           Test.cpp
           Rewriter.cpp
           )


target_include_directories(optimizer PRIVATE  ../include)

add_dependencies(optimizer OptimizerPassIncGen)


target_link_libraries(optimizer analysis)

target_precompile_headers(optimizer PRIVATE )
