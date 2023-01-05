include(GoogleTest)

add_executable(PatternTests
        PatternTests.cpp
        Pattern.cpp
)

llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(PatternTests lexer parser ${llvm_libs} ${GTest_LDFLAGS})

target_include_directories(PatternTests PUBLIC ../../code/parser ../../code/include ${GTest_INCLUDE_DIRS})

gtest_discover_tests(PatternTests)
