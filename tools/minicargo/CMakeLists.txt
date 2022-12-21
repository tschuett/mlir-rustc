add_executable(minicargo
           MiniCargo.cpp
           )


target_include_directories(minicargo PRIVATE  ../../include)


target_link_libraries(minicargo )
