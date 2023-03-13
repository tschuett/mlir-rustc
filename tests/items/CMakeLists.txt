include(GoogleTest)

add_executable(ItemsTests
        ItemTests.cpp
        Const.cpp
        Struct.cpp
        Implementation.cpp
        Function.cpp
        TypeAlias.cpp
)

llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(ItemsTests lexer parser adt ${llvm_libs} ${GTest_LDFLAGS})

target_include_directories(ItemsTests PUBLIC ../../code/parser ../../code/include ${GTest_INCLUDE_DIRS})

gtest_discover_tests(ItemsTests)
