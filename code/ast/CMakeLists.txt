add_library(ast
           OuterAttribute.cpp
           )


target_include_directories(ast PRIVATE  ../include)


target_link_libraries(ast )
