add_executable(rustc
           RustC.cpp
           )


target_include_directories(rustc PRIVATE  ../../code/include)


target_link_libraries(rustc lexer parser codegen)
