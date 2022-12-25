add_library(sema
           KeyWords.cpp
           Sema.cpp
           )

target_include_directories(sema PRIVATE  ../include)


target_link_libraries(sema )
