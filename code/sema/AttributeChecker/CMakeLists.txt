add_library(AttributeChecker
           AttributeChecker.cpp
           Items.cpp
           )

target_include_directories(AttributeChecker PRIVATE  ../../include)



target_link_libraries(AttributeChecker
                PRIVATE
)

