add_library(TypeChecking
            TypeChecking.cpp
            Item.cpp
            Function.cpp
            Types.cpp
            Expression.cpp
            Unification.cpp
            Closure.cpp
            Statement.cpp
            Pattern.cpp
            Literal.cpp
            Path.cpp
            Match.cpp
            Coercion.cpp
            PathProbing.cpp
            Pointer.cpp
            Substitutions.cpp
            Traits.cpp
            #Generics.cpp
            MethodProbe.cpp
            Trait.cpp
            Casting.cpp
            OperatorOverload.cpp
            MethodResolver.cpp
           )

target_include_directories(TypeChecking PRIVATE  ../../include)


target_link_libraries(TypeChecking )
