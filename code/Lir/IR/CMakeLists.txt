add_library(LirDialect
  LirDialect.cpp
  LirOps.cpp
  LirInterfaces.cpp
  LirTypes.cpp
  )


target_include_directories(LirDialect PRIVATE ../../include)

add_dependencies(LirDialect LirOpsIncGen)


target_link_libraries(LirDialect MLIRLLVMDialect MLIRIR)
