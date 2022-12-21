add_executable(rustc
           RustC.cpp
           )


target_include_directories(rustc PRIVATE  ../../include)


target_link_libraries(rustc lexer parser)
