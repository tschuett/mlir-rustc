add_library(Resolver
            Resolver.cpp
            NameResolution.cpp
           )

target_include_directories(Resolver PRIVATE  ../../include)


target_link_libraries(Resolver )
