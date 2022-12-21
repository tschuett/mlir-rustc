add_executable(minicargo
           MiniCargo.cpp
           )


target_include_directories(minicargo PRIVATE  ../../code/include)


target_link_libraries(minicargo toml)
