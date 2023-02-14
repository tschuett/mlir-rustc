add_library(parser
           Parser.cpp
           Util.cpp
           Function.cpp
           Attributes.cpp
           PatternNoTopAlt.cpp
           Expression.cpp
           Enumeration.cpp
           Closure.cpp
           Types.cpp
           Statement.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast lexer)

