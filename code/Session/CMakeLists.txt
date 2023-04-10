add_library(Session
           Session.cpp
           )

target_include_directories(Session PRIVATE  ../include)

target_link_libraries(
        Session
        PRIVATE
        )
