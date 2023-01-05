add_library(optimizer
           Test.cpp
           )


target_include_directories(optimizer PRIVATE  ../include)

add_dependencies(optimizer OptimizerPassIncGen)


target_link_libraries(optimizer)

target_precompile_headers(optimizer PRIVATE )
