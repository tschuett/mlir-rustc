add_library(parser
           Parser.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser )
