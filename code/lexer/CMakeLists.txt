add_library(lexer
           Lexer.cpp
           TokenStream.cpp
           Token.cpp
           KeyWords.cpp
           CodePoint.cpp
           Lexer2.cpp
           Identifier.cpp
           )


target_include_directories(lexer PRIVATE  ../include ${ICU_INCLUDE_DIR})

llvm_map_components_to_libnames(llvm_libs Support)


target_link_libraries(
        lexer
        PRIVATE
        adt
        ${llvm_libs}
        ${ICU_UC_LIBRARIES}
        )
