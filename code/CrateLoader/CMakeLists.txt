add_library(CrateLoader
           CrateLoader.cpp
           LoadModule.cpp
           )

target_include_directories(CrateLoader PRIVATE  ../include)


target_link_libraries(CrateLoader ast Serialization lexer parser Mappings)
