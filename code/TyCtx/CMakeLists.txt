add_library(TyCtx
           TyCtx.cpp
           TyTy.cpp
           TraitReference.cpp
           Substitutions.cpp
           SubstitutionsMapper.cpp
           AssociatedImplTrait.cpp
           Bounds.cpp
           Unification.cpp
           )

target_include_directories(TyCtx PRIVATE  ../include)

llvm_map_components_to_libnames(llvm_libs support)

target_link_libraries(
        TyCtx
        PRIVATE
        Basic
        Session
        adt
        ast
        lexer
        ${llvm_libs}
        )
