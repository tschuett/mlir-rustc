add_library(Resolver
            Resolver.cpp
            Function.cpp
            NoRecurse.cpp
            Expression.cpp
            Types.cpp
            Generics.cpp
            Patterns.cpp
            Items.cpp
            Statements.cpp
            Builtins.cpp
           )

target_include_directories(Resolver PRIVATE  ../../include)


target_link_libraries(Resolver adt)
