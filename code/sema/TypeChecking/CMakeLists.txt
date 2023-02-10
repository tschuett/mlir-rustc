add_library(TypeChecking
            TypeChecking.cpp
           )

target_include_directories(TypeChecking PRIVATE  ../../include)


target_link_libraries(TypeChecking )
