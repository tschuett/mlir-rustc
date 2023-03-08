add_library(Resolver
            Resolver.cpp
            Function.cpp
            NoRecurse.cpp
           )

target_include_directories(Resolver PRIVATE  ../../include)


target_link_libraries(Resolver adt)
