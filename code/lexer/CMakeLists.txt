add_library(lexer
           Lexer.cpp
           )


target_include_directories(lexer PRIVATE  ../include)


target_link_libraries(lexer )
