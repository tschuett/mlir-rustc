add_library(lexer
           Lexer.cpp
           TokenStream.cpp
           Token.cpp
           KeyWords.cpp
           CodePoint.cpp
           )


target_include_directories(lexer PRIVATE  ../include)


target_link_libraries(lexer )
