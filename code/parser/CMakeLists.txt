add_library(parser
           Parser.cpp
           Attributes.cpp
           UseDeclaration.cpp
           Visibility.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast)
