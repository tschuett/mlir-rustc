add_library(MirDialect
  MirDialect.cpp
  MirOps.cpp
  MirInterfaces.cpp
  )


target_include_directories(MirDialect PRIVATE ../../include)

add_dependencies(MirDialect MirOpsIncGen MirInterfacesIncGen)


target_link_libraries(MirDialect MLIRLLVMDialect MLIRIR)


