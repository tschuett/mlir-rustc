add_library(AttributeChecker
           AttributeChecker.cpp
           Items.cpp
           )

target_include_directories(AttributeChecker PRIVATE  ../../include)

llvm_map_components_to_libnames(llvm_libs support)



target_link_libraries(AttributeChecker
                PRIVATE
                ast
                ${llvm_libs}
)

