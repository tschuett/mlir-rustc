add_library(CrateLoader
           CrateLoader.cpp
           LoadModule.cpp
           )

target_include_directories(CrateLoader PRIVATE  ../include)


target_link_libraries(CrateLoader sema Serialization lexer parser)
