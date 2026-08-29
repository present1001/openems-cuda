# Currently, GCC generates a massive warning with over 60 lines
# for a single class, the last few lines are:
# 
#     fpoptimizer.cc:768:13: warning: implicitly-declared
#     ‘FPoptimizer_Optimize::MatchInfo<double>::MatchInfo(const
#     FPoptimizer_Optimize::MatchInfo<double>&)’ is deprecated
#     [-Wdeprecated-copy]
#       768 | #define yD1 AnyParams_Rec
#           |             ^~~~~~~~~~~~~
# 
# This is a legitimate warning about the potential removal of a
# deprecated default behavior by future C++ versions.
# 
# Since fparser is already a legacy project, and the problem exists
# in auto-generates code, fixing it is a very-low priority task. For
# now, restrict ourselves to C++98 only, removing the warning and
# ensuring its behavior won't change for any future compiler.
# Optionally, fparser has C++11 features controlled via
# FP_SUPPORT_CPLUSPLUS11_MATH_FUNCS, but none is enabled in this fork.
set(CMAKE_CXX_STANDARD 98)

# define build type
SET( CMAKE_BUILD_TYPE Release )

# In CMake 4, 3.10 is deprecated and 3.5 has been removed.
# use 3.1...3.10 so all of these versions are acceptable as min. version.
# https://cmake.org/cmake/help/latest/command/cmake_minimum_required.html
cmake_minimum_required(VERSION 3.1...3.10)

PROJECT(fparser CXX)

# default
set(LIB_VERSION_MAJOR 4)
set(LIB_VERSION_MINOR 5)
set(LIB_VERSION_PATCH 1)
set(LIB_VERSION_STRING ${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH})

set(VERSION "v${LIB_VERSION_STRING}")

if(APPLE AND NOT CMAKE_INSTALL_NAME_DIR)
  set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
endif()

#set(HEADERS fparser.hh fpconfig.hh fptypes.hh)
set(SOURCES fparser.cc fpoptimizer.cc)

remove(CMAKE_CXX_FLAGS -Dfparser_EXPORTS)

ADD_LIBRARY(fparser SHARED ${SOURCES})

set_target_properties(fparser PROPERTIES VERSION ${LIB_VERSION_STRING} SOVERSION ${LIB_VERSION_MAJOR})

if (WIN32)
    INSTALL(TARGETS fparser RUNTIME DESTINATION bin ARCHIVE DESTINATION lib${LIB_SUFFIX})
else()
    INSTALL(TARGETS fparser DESTINATION lib${LIB_SUFFIX})
endif()

INSTALL(FILES fparser.hh DESTINATION include)

