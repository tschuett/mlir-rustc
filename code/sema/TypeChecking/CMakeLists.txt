add_library(TypeChecking
            TypeChecking.cpp
            Item.cpp
            Function.cpp
            Types.cpp
            Expression.cpp
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
            Resolver.cpp
            TypeBoundsProbe.cpp
            TraitResolver.cpp
            Call.cpp
           )

target_include_directories(TypeChecking PRIVATE  ../../include)

llvm_map_components_to_libnames(llvm_libs support)


target_link_libraries(
        TypeChecking
        PRIVATE
        ${llvm_libs}
        TyCtx
        )
