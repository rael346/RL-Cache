cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# export compile_commands.json (for clangd lsp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

project(main LANGUAGES CXX)

# set(CMAKE_PREFIX_PATH /Users/duytran/mamba/envs/rl-cache)

# executables
add_executable(
	feature_collector
	feature_collector/main.cpp
	feature_collector/FeatureCollector.cpp
	feature_collector/InfoCollector.cpp
	feature_collector/Environment.cpp
)

add_executable(
	reward_collector
	reward_collector/main.cpp
	reward_collector/FeatureCollector.cpp
	reward_collector/Environment.cpp
)

find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter Development)
find_package(Boost 1.89 CONFIG REQUIRED COMPONENTS python313)
# if (Python3_FOUND)
# 	message(STATUS "Found Python 3: ${Python3_Interpreter_FOUND}")
# 	message(STATUS "Python 3 Version: ${Python3_VERSION}")
# 	message(STATUS "Python 3 Libraries: ${Python3_LIBRARIES}")
# 	message(STATUS "Python 3 Include Dirs: ${Python3_INCLUDE_DIRS}")
# else()
# 	message(FATAL_ERROR "Python 3 not found. Please install Python 3 development files.")
# endif()
#
# message(STATUS "boost include ${Boost_INCLUDE_DIRS}")
# message(STATUS "boost lib ${Boost_LIBRARIES_DIRS}")

# set(Python3_INCLUDE_DIRS /Users/duytran/mamba/envs/rl-cache/include/python3.12)
# set(Python3_LIBRARIES /Users/duytran/mamba/envs/rl-cache/lib/libpython3.12.dylib)
#
# set(Boost_INCLUDE_DIRS /Users/duytran/mamba/envs/rl-cache/include)
# set(Boost_LIBRARIES /Users/duytran/mamba/envs/rl-cache/lib/libboost_python312.dylib)

add_library(LRUSim MODULE
	caching_classes/LRUSim.cpp
	caching_classes/CacheSim.cpp
)
target_include_directories(LRUSim PRIVATE ${Python3_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(LRUSim PRIVATE ${Python3_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(LRUSim PROPERTIES PREFIX "")
set_target_properties(LRUSim PROPERTIES
				LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_library(AdaptSizeSim MODULE
	caching_classes/AdaptSizeSim.cpp
	caching_classes/CacheSim.cpp
)
target_include_directories(AdaptSizeSim PRIVATE ${Python3_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(AdaptSizeSim PRIVATE ${Python3_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(AdaptSizeSim PROPERTIES PREFIX "")
set_target_properties(AdaptSizeSim PROPERTIES
				LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
