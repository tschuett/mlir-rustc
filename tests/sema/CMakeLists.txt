include(GoogleTest)

add_executable(SemaTests
        SemaTests.cpp
        Function.cpp
)

llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(SemaTests
        PRIVATE
        lexer
        parser
        sema
        adt
        ${llvm_libs}
        GTest::gtest
        GTest::gtest_main
        )

target_include_directories(SemaTests PUBLIC ../../code/include ${GTEST_INCLUDE_DIRS})

gtest_discover_tests(SemaTests)
