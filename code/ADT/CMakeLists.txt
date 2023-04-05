add_library(adt
           ScopedCanonicalPath.cpp
           Utf8String.cpp
           CanonicalPath.cpp
           )

target_include_directories(adt PRIVATE  ../include ${ICU_INCLUDE_DIR})

target_link_libraries(
        adt
        PRIVATE
        ${ICU_UC_LIBRARIES}
        )
