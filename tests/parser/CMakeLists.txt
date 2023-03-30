include(GoogleTest)

add_executable(ParserTests
        ParserTests.cpp
        Parser.cpp
)

llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(ParserTests lexer parser adt ${llvm_libs} GTest::gtest GTest::gtest_main)

target_include_directories(ParserTests PUBLIC ../../code/parser ../../code/include ${GTEST_INCLUDE_DIRS})

gtest_discover_tests(ParserTests)
