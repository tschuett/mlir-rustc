add_library(ConstantEvaluation
           ConstantEvaluation.cpp
           Owner.cpp
           )

target_include_directories(ConstantEvaluation PRIVATE  ../include)

target_link_libraries(
        ConstantEvaluation
        PRIVATE
        )
