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
include CMakeFiles/ad-census.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ad-census.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ad-census.dir/flags.make

CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o: CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o.depend
CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o: CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o.Debug.cmake
CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o: ../cuda_c_ch1/main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o"
	cd /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/cuda_c_ch1 && /data/private/wl2020/workspace/cmake-3.15.0-Linux-x86_64/bin/cmake -E make_directory /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/cuda_c_ch1/.
	cd /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/cuda_c_ch1 && /data/private/wl2020/workspace/cmake-3.15.0-Linux-x86_64/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/cuda_c_ch1/./ad-census_generated_main.cu.o -D generated_cubin_file:STRING=/home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/cuda_c_ch1/./ad-census_generated_main.cu.o.cubin.txt -P /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o.Debug.cmake

# Object files for target ad-census
ad__census_OBJECTS =

# External object files for target ad-census
ad__census_EXTERNAL_OBJECTS = \
"/home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o"

ad-census: CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o
ad-census: CMakeFiles/ad-census.dir/build.make
ad-census: /usr/local/cuda-10.1/lib64/libcudart_static.a
ad-census: /usr/lib/x86_64-linux-gnu/librt.so
ad-census: CMakeFiles/ad-census.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ad-census"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ad-census.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ad-census.dir/build: ad-census

.PHONY : CMakeFiles/ad-census.dir/build

CMakeFiles/ad-census.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ad-census.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ad-census.dir/clean

CMakeFiles/ad-census.dir/depend: CMakeFiles/ad-census.dir/cuda_c_ch1/ad-census_generated_main.cu.o
	cd /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wl2020/workspace/cuda_work /home/wl2020/workspace/cuda_work /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu /home/wl2020/workspace/cuda_work/cmake-build-debug-blcu/CMakeFiles/ad-census.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ad-census.dir/depend

