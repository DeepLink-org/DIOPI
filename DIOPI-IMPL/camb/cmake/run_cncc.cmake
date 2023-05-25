#  This code is licensed under the MIT License.  See the FindBANG.cmake script
#  for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


##########################################################################
# This file runs the cncc commands to produce the desired output file along with
# the dependency file needed by CMake to compute dependencies.  In addition the
# file checks the output of each command and if the command fails it deletes the
# output files.

# Input variables
#
# verbose:BOOL=<>          OFF: Be as quiet as possible (default)
#                          ON : Describe each step
#
# build_configuration:STRING=<> Typically one of Debug, MinSizeRel, Release, or
#                               RelWithDebInfo, but it should match one of the
#                               entries in BANG_HOST_FLAGS. This is the build
#                               configuration used when compiling the code.  If
#                               blank or unspecified Debug is assumed as this is
#                               what CMake does.
#
# generated_file:STRING=<> File to generate.  This argument must be passed in.
#
# generated_cnbin_file:STRING=<> File to generate.  This argument must be passed
#                                                   in if build_cnbin is true.

if(NOT generated_file)
  message(FATAL_ERROR "You must specify generated_file on the command line")
endif()

# Set these up as variables to make reading the generated file easier
set(CMAKE_COMMAND "@CMAKE_COMMAND@") # path
set(source_file "@source_file@") # path
set(CNCC_generated_dependency_file "@CNCC_generated_dependency_file@") # path
set(cmake_dependency_file "@cmake_dependency_file@") # path
set(BANG_make2cmake "@BANG_make2cmake@") # path
set(BANG_parse_cnbin "@BANG_parse_cnbin@") # path
set(build_cnbin @build_cnbin@) # bool
set(BANG_HOST_COMPILER "@BANG_HOST_COMPILER@") # path
# We won't actually use these variables for now, but we need to set this, in
# order to force this file to be run again if it changes.
set(generated_file_path "@generated_file_path@") # path
set(generated_file_internal "@generated_file@") # path
set(generated_cnbin_file_internal "@generated_cnbin_file@") # path

set(BANG_CNCC_EXECUTABLE "@BANG_CNCC_EXECUTABLE@") # path
set(BANG_CNCC_FLAGS @BANG_CNCC_FLAGS@ ;; @BANG_WRAP_OPTION_CNCC_FLAGS@) # list
@BANG_CNCC_FLAGS_CONFIG@
set(cncc_flags @cncc_flags@) # list
set(BANG_CNCC_INCLUDE_ARGS "@BANG_CNCC_INCLUDE_ARGS@") # list (needs to be in quotes to handle spaces properly).
set(format_flag "@format_flag@") # string
set(bang_language_flag @bang_language_flag@) # list
set(BANG_GENERATE_INCLUDES_DEPENDENCIES @BANG_GENERATE_INCLUDES_DEPENDENCIES@) # bool

if(build_cnbin AND NOT generated_cnbin_file)
  message(FATAL_ERROR "You must specify generated_cnbin_file on the command line")
endif()

# This is the list of host compilation flags.  It C or CXX should already have
# been chosen by FindBANG.cmake.
@BANG_HOST_FLAGS@

# Take the compiler flags and package them up to be sent to the compiler via -Xcompiler
set(cncc_host_compiler_flags "")
# If we weren't given a build_configuration, use Debug.
if(NOT build_configuration)
  set(build_configuration Debug)
endif()
string(TOUPPER "${build_configuration}" build_configuration)
#message("BANG_CNCC_HOST_COMPILER_FLAGS = ${BANG_CNCC_HOST_COMPILER_FLAGS}")
foreach(flag ${CMAKE_HOST_FLAGS} ${CMAKE_HOST_FLAGS_${build_configuration}})
  # Extra quotes are added around each flag to help cncc parse out flags with spaces.
  set(cncc_host_compiler_flags ${cncc_host_compiler_flags} ${flag})
endforeach()
# message("cncc_host_compiler_flags = ${cncc_host_compiler_flags}")
# Add the build specific configuration flags
list(APPEND BANG_CNCC_FLAGS ${BANG_CNCC_FLAGS_${build_configuration}})

# bang_execute_process - Executes a command with optional command echo and status message.
#
#   status  - Status message to print if verbose is true
#   command - COMMAND argument from the usual execute_process argument structure
#   ARGN    - Remaining arguments are the command with arguments
#
#   BANG_result - return value from running the command
#
# Make this a macro instead of a function, so that things like RESULT_VARIABLE
# and other return variables are present after executing the process.
macro(bang_execute_process status command)
  set(_command ${command})
  if(NOT "x${_command}" STREQUAL "xCOMMAND")
    message(FATAL_ERROR "Malformed call to bang_execute_process.  Missing COMMAND as second argument. (command = ${command})")
  endif()
  if(verbose)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E echo -- ${status})
    # Now we need to build up our command string.  We are accounting for quotes
    # and spaces, anything else is left up to the user to fix if they want to
    # copy and paste a runnable command line.
    set(bang_execute_process_string)
    foreach(arg ${ARGN})
      # If there are quotes, excape them, so they come through.
      string(REPLACE "\"" "\\\"" arg ${arg})
      # Args with spaces need quotes around them to get them to be parsed as a single argument.
      if(arg MATCHES " ")
        list(APPEND bang_execute_process_string "\"${arg}\"")
      else()
        list(APPEND bang_execute_process_string ${arg})
      endif()
    endforeach()
    # Echo the command
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${bang_execute_process_string})
  endif()
  # Run the command
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE BANG_result )
endmacro()

# Delete the target file
bang_execute_process(
  "Removing ${generated_file}"
  COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
  )

# cncc ignore host flags
set(cncc_host_compiler_flags "")

if (BANG_GENERATE_INCLUDES_DEPENDENCIES)
# use '-M' to get header depedencies
set(preprocess_cflags ${BANG_CNCC_FLAGS})
list(REMOVE_ITEM preprocess_cflags -Wall -Werror)
set(_cmd_args "${BANG_CNCC_EXECUTABLE}" "${source_file}" ${bang_language_flag} ${cncc_flags} ${preprocess_cflags} -Wno-unused-command-line-argument -DCNCC ${BANG_CNCC_INCLUDE_ARGS} -Wno-error -MM -MT ${generated_file})
if (verbose)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${_cmd_args})
endif()
execute_process(
  COMMAND ${_cmd_args}
  COMMAND sed -z "s|\\\\\\n||g"
  OUTPUT_VARIABLE source_dep_rules
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
#message("source_dep_rules: ${source_dep_rules}")
execute_process(
  COMMAND echo "${source_dep_rules}"
  COMMAND grep ":"
  COMMAND cut -d ":" -f1
  COMMAND sort
  COMMAND uniq
  COMMAND sed -z "s/\\n/;/g"
  OUTPUT_VARIABLE source_dep_keys
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
#message("source_dep_keys: ${source_dep_keys}")

set(content)
# convert multiple dependencies (based on different mlu arch) into single one
foreach(key ${source_dep_keys})
  #message("proc key ${key}")
  execute_process(
    COMMAND echo "${source_dep_rules}"
    COMMAND grep "^${key}:"
    COMMAND cut -d ":" -f2-
    COMMAND tr " " "\\n"
    COMMAND sort
    COMMAND uniq
    COMMAND grep .
    COMMAND sed -z "s/\\n/ /g"
    OUTPUT_VARIABLE dep_files
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  #message("Add key: ${key}, content: ${dep_files}")
  set(content "${content}${key}: ${dep_files}\n")
endforeach()
#message("content: ${content}")
file(WRITE "${generated_file}.d" "${content}")
if (verbose)
  message("Generated dependency file ${generated_file}.d successfully.")
endif()
endif() # BANG_GENERATE_INCLUDES_DEPENDENCIES

# Generate the code
bang_execute_process(
  "Generating ${generated_file}"
  COMMAND "${BANG_CNCC_EXECUTABLE}"
  "${source_file}"
  ${bang_language_flag}
  ${format_flag} -o "${generated_file}"
  ${cncc_flags}
  ${cncc_host_compiler_flags}
  ${BANG_CNCC_FLAGS}
  -DCNCC
  ${BANG_CNCC_INCLUDE_ARGS}
  )

if(BANG_result)
  # Since cncc can sometimes leave half done files make sure that we delete the output file.
  bang_execute_process(
    "Removing ${generated_file}"
    COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
    )
  message(FATAL_ERROR "Error generating file ${generated_file}")
else()
  if(verbose)
    message("Generated ${generated_file} successfully.")
  endif()
endif()

# Cnbin resource report commands.
if( build_cnbin )
  # Run with -cnbin to produce resource usage report.
  bang_execute_process(
    "Generating ${generated_cnbin_file}"
    COMMAND "${BANG_CNCC_EXECUTABLE}"
    "${source_file}"
    ${BANG_CNCC_FLAGS}
    ${cncc_flags}
    ${cncc_host_compiler_flags}
    -DCNCC
    -cnbin
    -o "${generated_cnbin_file}"
    ${BANG_CNCC_INCLUDE_ARGS}
    )

  # Execute the parser script.
  bang_execute_process(
    "Executing the parser script"
    COMMAND  "${CMAKE_COMMAND}"
    -D "input_file:STRING=${generated_cnbin_file}"
    -P "${BANG_parse_cnbin}"
    )

endif()
