add_library(Mappings
           Mappings.cpp
           )


target_include_directories(Mappings PRIVATE  ../include)

target_link_libraries(Mappings )
