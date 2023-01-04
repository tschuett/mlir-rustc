add_library(sema
           Sema.cpp
           )

target_include_directories(sema PRIVATE  ../include)


target_link_libraries(sema )
