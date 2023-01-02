#add_mlir_dialect_library(Mir
#  MirDialect.cpp
#  DEPENDS
##  MirOpsIncGen
##  MirTransformsIncGen
#
#  LINK_COMPONENTS
#  Core
#
#  LINK_LIBS PUBLIC
##  MLIRBar
##  <some-other-library>
#  )


add_library(MirDialect
  MirDialect.cpp
  MirOps.cpp
  )

set_target_properties(MirDialect PROPERTIES
        CXX_EXCEPTIONS OFF
        CXX_RTTI OFF
    )

target_include_directories(MirDialect PRIVATE ../../include)

add_dependencies(MirDialect MirOpsIncGen ) #MirOpInterfacesIncGen

target_link_libraries(MirDialect MLIRLLVMDialect)


