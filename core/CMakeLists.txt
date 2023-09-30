set(KERNEL_CPP)

file(GLOB_RECURSE TENSOR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.h")
message(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE KERNEL_CPP "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cc")
message("dadad")

# https://stackoverflow.com/questions/11096471/how-can-i-install-a-hierarchy-of-files-using-cmake
set(install_headers ${TENSOR_HEADERS})
set(cpu_cpp ${KERNEL_CPP})

foreach(item IN LISTS cpu_cpp)
    message("Item: ${item}")
endforeach()

#[[
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/core/tensor)
foreach ( HEADER   ${install_headers} )
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "core/" HEADER_SUB ${HEADER})
    string(REPLACE "${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/" "" HEADER_SUB ${HEADER_SUB})
    get_filename_component( dir ${HEADER_SUB} DIRECTORY )
    message(${dir})
    message(include/${dir})
    message(----------------------)
    message(${HEADER})
    message(${CMAKE_BINARY_DIR}/include/${dir})
    install( FILES ${HEADER} DESTINATION  ${CMAKE_BINARY_DIR}/include/${dir} )
endforeach()
]]
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(ops ${cpu_cpp})
add_executable(broadcast kernels/broadcast_in_dim.cc)
target_link_libraries(broadcast ops)

# 原因在于上面的include的文件并没有创建
message(${CMAKE_CURRENT_SOURCE_DIR}/tensor)

#[[

]]
