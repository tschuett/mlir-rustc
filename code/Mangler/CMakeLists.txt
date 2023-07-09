add_library(Mangler
           Mangler.cpp
           )

target_include_directories(Mangler PRIVATE  ../include)

target_link_libraries(
        Mangler
        PRIVATE
        ConstantEvaluation
        )
