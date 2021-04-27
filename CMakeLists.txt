cmake_minimum_required(VERSION 3.0)
project(ddc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

find_package(Armadillo CONFIG REQUIRED)
find_package(GSL REQUIRED)


add_library(ddc SHARED
        include/psi.hpp
        include/basic_stats.hpp
        include/regression.hpp
        include/robust_correlation.hpp
        include/robust_estimators.hpp
        src/robust_correlation.cpp
        src/c_exports.cpp src/robust_estimators.cpp src/regression.cpp src/univariate.cpp src/ddc.cpp)

# add_subdirectory(cpp)

target_include_directories(ddc PUBLIC include ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(ddc PUBLIC ${ARMADILLO_LIBRARIES}  ${GSL_LIBRARIES})
