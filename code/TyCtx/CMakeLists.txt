add_library(TyCtx
           TyCtx.cpp
           TyTy.cpp
           TraitReference.cpp
           Substitutions.cpp
           SubstitutionsMapper.cpp
           AssociatedImplTrait.cpp
           Bounds.cpp
           )

target_include_directories(TyCtx PRIVATE  ../include)

target_link_libraries(
        TyCtx
        PRIVATE
        Basic
        )
