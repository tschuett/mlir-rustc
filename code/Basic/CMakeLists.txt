add_library(Basic
           Basic.cpp
           Ids.cpp
           )


target_include_directories(Basic PRIVATE  ../include)


target_link_libraries(Basic )

