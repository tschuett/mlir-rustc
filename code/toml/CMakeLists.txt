add_library(toml
           Toml.cpp
           TomlLexer.cpp
           TokenStream.cpp
           Parser.cpp
           Table.cpp
           )


target_include_directories(toml PRIVATE  ../include)


target_link_libraries(toml )
