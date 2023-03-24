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
            Path.cpp
            Closure.cpp
            Match.cpp
            Loops.cpp
            Struct.cpp
           )

target_include_directories(Resolver PRIVATE  ../../include)


target_link_libraries(Resolver
                PRIVATE
                adt
                TyCtx
)
