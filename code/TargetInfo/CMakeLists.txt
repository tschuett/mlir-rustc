add_library(TargetInfo
           TargetInfo.cpp
           )


target_include_directories(TargetInfo PRIVATE  ../include)


#llvm_map_components_to_libnames(llvm_libs TargetParser)

target_link_libraries(TargetInfo  ${llvm_libs} )


#target_precompile_headers(modulebuilder PRIVATE ../include/ModuleBuilder/ModuleBuilder.h)



add_subdirectory(Linux)
