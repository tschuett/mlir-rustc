include(GoogleTest)

add_executable(ADTTests
        ADTTests.cpp
        ScopedHashTable.cpp
)

llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(ADTTests ${llvm_libs} GTest::gtest GTest::gtest_main)

target_include_directories(ADTTests PUBLIC ../../code/include ${GTEST_INCLUDE_DIRS})

gtest_discover_tests(ADTTests)
