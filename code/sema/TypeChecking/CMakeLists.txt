add_library(TypeChecking
            TypeChecking.cpp
            TyTy.cpp
            Item.cpp
            Function.cpp
            Types.cpp
            Expression.cpp
           )

target_include_directories(TypeChecking PRIVATE  ../../include)


target_link_libraries(TypeChecking )
