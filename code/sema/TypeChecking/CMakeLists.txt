add_library(TypeChecking
            TypeChecking.cpp
            TyTy.cpp
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
            Substitutions.cpp
            Coercion.cpp
            PathProbing.cpp
           )

target_include_directories(TypeChecking PRIVATE  ../../include)


target_link_libraries(TypeChecking )
