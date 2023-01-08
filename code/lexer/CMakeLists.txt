add_library(lexer
           Lexer.cpp
           TokenStream.cpp
           Token.cpp
           KeyWords.cpp
           CodePoint.cpp
           )


target_include_directories(lexer PRIVATE  ../include)

llvm_map_components_to_libnames(llvm_libs Support)


target_link_libraries(lexer ${llvm_libs})
