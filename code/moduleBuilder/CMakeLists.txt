add_library(modulebuilder
           ModuleBuilder.cpp
           Statements.cpp
           Expressions.cpp
           ExpressionWithBlock.cpp
           ExpressionWithoutBlock.cpp
           Target.cpp
           Function.cpp
           Item.cpp
           OperatorExpression.cpp
           Types.cpp
           Expression.cpp
           LiteralExpression.cpp
           PathExpression.cpp
           BlockExpression.cpp
           Declare.cpp
           )


target_include_directories(modulebuilder PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs TargetParser)

target_link_libraries(modulebuilder remarks MirDialect ast ${llvm_libs} MLIRFuncDialect MLIRAsyncDialect MLIRMemRefDialect)


target_precompile_headers(modulebuilder PRIVATE ../include/ModuleBuilder/ModuleBuilder.h)

