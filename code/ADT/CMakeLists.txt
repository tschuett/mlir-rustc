add_library(adt
           ScopedCanonicalPath.cpp
           )

target_include_directories(adt PRIVATE  ../include)


target_link_libraries(adt )
