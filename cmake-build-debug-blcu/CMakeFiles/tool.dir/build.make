# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/wl2020/workspace/cmake-3.15.0-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/wl2020/workspace/cmake-3.15.0-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wl2020/workspace/cuda_work

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu

# Include any dependencies generated for this target.
include CMakeFiles/tool.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tool.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tool.dir/flags.make

CMakeFiles/tool.dir/tool.cpp.o: CMakeFiles/tool.dir/flags.make
CMakeFiles/tool.dir/tool.cpp.o: ../tool.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tool.dir/tool.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tool.dir/tool.cpp.o -c /home/wl2020/workspace/cuda_work/tool.cpp

CMakeFiles/tool.dir/tool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tool.dir/tool.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wl2020/workspace/cuda_work/tool.cpp > CMakeFiles/tool.dir/tool.cpp.i

CMakeFiles/tool.dir/tool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tool.dir/tool.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wl2020/workspace/cuda_work/tool.cpp -o CMakeFiles/tool.dir/tool.cpp.s

# Object files for target tool
tool_OBJECTS = \
"CMakeFiles/tool.dir/tool.cpp.o"

# External object files for target tool
tool_EXTERNAL_OBJECTS =

tool: CMakeFiles/tool.dir/tool.cpp.o
tool: CMakeFiles/tool.dir/build.make
tool: CMakeFiles/tool.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tool"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tool.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tool.dir/build: tool

.PHONY : CMakeFiles/tool.dir/build

CMakeFiles/tool.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tool.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tool.dir/clean

CMakeFiles/tool.dir/depend:
	cd /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wl2020/workspace/cuda_work /home/wl2020/workspace/cuda_work /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/tool.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tool.dir/depend

