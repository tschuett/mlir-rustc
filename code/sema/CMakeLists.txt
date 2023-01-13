add_library(sema
           Sema.cpp
           Module.cpp
           Item.cpp
           VisItem.cpp
           OuterAttributes.cpp
           )

target_include_directories(sema PRIVATE  ../include)


target_link_libraries(sema )
