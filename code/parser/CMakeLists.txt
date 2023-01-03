add_library(parser
           Parser.cpp
           Attributes.cpp
           UseDeclaration.cpp
           Visibility.cpp
           Modules.cpp
           Item.cpp
           Function.cpp
           SimplePath.cpp
           Util.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast)

target_precompile_headers(parser PRIVATE Modules.h Visibility.h Attributes.h Item.h Function.h UseDeclaration.h SimplePath.h Util.h)
