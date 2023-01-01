add_library(modulebuilder
           ModuleBuilder.cpp
           Statements.cpp
           Expressions.cpp
           Target.cpp
           Function.cpp
           )


target_include_directories(modulebuilder PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs TargetParser)

target_link_libraries(modulebuilder MirDialect ast ${llvm_libs} MLIRFuncDialect MLIRAsyncDialect)


target_precompile_headers(modulebuilder PRIVATE llvm/Target/TargetMachine.h mlir/IR/MLIRContext.h)

