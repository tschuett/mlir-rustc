add_library(crate_builder
           CrateBuilder.cpp
           Item.cpp
           Function.cpp
           Module.cpp
           BlockExpression.cpp
           Statements.cpp
           ExpressionWithoutBlock.cpp
           ExpressionWithBlock.cpp
           Statement.cpp
           ExpressionStatement.cpp
           LoopExpression.cpp
           Expression.cpp
           OperatorExpression.cpp
           Target.cpp
           Calls.cpp
           Types.cpp
           LetStatement.cpp
           ComparePatternWithExpression.cpp
           MatchIfLetPattern.cpp
           ConstantEvaluator.cpp
           Owner.cpp
           Struct.cpp
           )


target_include_directories(crate_builder PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs Remarks)

target_link_libraries(crate_builder
        PRIVATE
        ast
        hir
        adt
        Mangler
        ConstantEvaluation
        ${llvm_libs}
        MLIRMemRefDialect
        MLIRArithDialect
        MLIRFuncDialect
        MLIRVectorDialect
)


target_precompile_headers(crate_builder PRIVATE )
