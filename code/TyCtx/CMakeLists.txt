add_library(TyCtx
           TyCtx.cpp
           TyTy.cpp
           Substitutions.cpp
           )

target_include_directories(TyCtx PRIVATE  ../include)

target_link_libraries(
        TyCtx
        PRIVATE
        Basic
        )
