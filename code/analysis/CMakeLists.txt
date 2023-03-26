add_library(analysis
           MemorySSA.cpp
           AliasAnalysis.cpp
           Attributer.cpp
           DependencyGraph.cpp
           DependencyGraphNode.cpp
           AANoop.cpp
           MemorySSANodes.cpp
           Abstractelement.cpp
           Loops.cpp
           PointsTo.cpp
#           Cycles.cpp
           ScalarEvolution.cpp
           )

target_include_directories(analysis PRIVATE  ../include)

llvm_map_components_to_libnames(llvm_libs)


target_link_libraries(analysis
        PRIVATE
        ${llvm_libs}
        )
