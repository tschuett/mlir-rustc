include(GoogleTest)

add_executable(TypesTests
        TypesTests.cpp
        Types.cpp
)

llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(TypesTests lexer parser ${llvm_libs} ${GTest_LDFLAGS})

target_include_directories(TypesTests PUBLIC ../../code/parser ../../code/include ${GTest_INCLUDE_DIRS})

gtest_discover_tests(TypesTests)
