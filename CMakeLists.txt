project(fora)
# Specify the minimum version for CMake
cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall")
#SET(CMAKE_CXX_FLAGS "-std=c++14 -O3 -DNDEBUG -w -pthread -lboost_serialization -lboost_filesystem -lboost_system")
SET(CMAKE_CXX_FLAGS "-std=c++14 -O3 -w -pthread -lboost_serialization -fopenmp -lboost_filesystem -lboost_system -lboost_date_time")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
SET(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym") 
SET(CMAKE_BUILD_TYPE DEBUG)
#SET(CMAKE_CXX_FLAGS "-std=c++14 -O3 -w -pthread")


find_package(Boost REQUIRED filesystem system date_time program_options)
FIND_PACKAGE( OpenMP)
  if(OPENMP_FOUND)
     message("OPENMP FOUND")
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
 include_directories(SYSTEM ${OpenMP_INCLUDE_PATH})
#find_package(Threads)

#set(HEADER algo.h build_oracle.h config.h graph.h head.h heap.h mylib.h query.h)
#set(PPRLIB  mylib.cpp config.cpp)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")


#add_library(mylib ${PROJECT_SOURCE_DIR}/mylib.cpp ${PROJECT_SOURCE_DIR}/mylib.h)
#add_library(config ${PROJECT_SOURCE_DIR}/config.cpp ${PROJECT_SOURCE_DIR}/config.h)

add_executable(fora ${PROJECT_SOURCE_DIR}/fora.cpp ${PROJECT_SOURCE_DIR}/mylib.cpp ${PROJECT_SOURCE_DIR}/config.cpp)

target_link_libraries(fora pthread boost_serialization boost_filesystem boost_system boost_date_time ${OpenMP_CXX_LIBRARIES})
