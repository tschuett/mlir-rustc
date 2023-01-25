add_library(ast2hir
        Lower.cpp
           )


target_include_directories(ast2hir PRIVATE  ../include)


target_link_libraries(ast2hir ast hir)
