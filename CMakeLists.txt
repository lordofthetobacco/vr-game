cmake_minimum_required(VERSION 4.2)
project(ForceUnleashedVR VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-Wpedantic>"
)
# Per-target compile options must be set after the target is created.

find_package(SDL3 REQUIRED)
if(SDL3_FOUND)
    message(STATUS "Found SDL3: ${SDL3_VERSION_STRING}")
else()
    message(FATAL_ERROR "SDL3 not found")
endif()

find_package(Vulkan REQUIRED)
if(Vulkan_FOUND)
    message(STATUS "Found Vulkan: ${Vulkan_VERSION}")
else()
    message(FATAL_ERROR "Vulkan not found")
endif()

find_package(assimp REQUIRED)
if(assimp_FOUND)
    message(STATUS "Found Assimp: ${ASSIMP_VERSION}")
else()
    message(FATAL_ERROR "Assimp not found")
endif()

add_executable(ForceUnleashedVR
    src/main.cpp
    src/io.cpp
    src/model.cpp
    src/object.cpp
    src/renderer.cpp
    src/stb_image_impl.cpp
)
target_compile_options(ForceUnleashedVR PRIVATE
    "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-Wpedantic>"
)
target_include_directories(ForceUnleashedVR PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ForceUnleashedVR PRIVATE SDL3::SDL3 Vulkan::Vulkan assimp::assimp)

# stb_image is third-party single-header code: suppress its warnings.
set_source_files_properties(src/stb_image_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")

# --- Shader compilation (GLSL -> SPIR-V) ---
find_program(GLSLC NAMES glslc
    HINTS
        "$ENV{VULKAN_SDK}/bin"
        "/home/fcieslar/Downloads/1.4.350.1/x86_64/bin"
)
if(NOT GLSLC)
    message(FATAL_ERROR
        "glslc not found. Source the Vulkan SDK setup-env.sh or set VULKAN_SDK.")
endif()
message(STATUS "Found glslc: ${GLSLC}")

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SHADER_SOURCES
    ${CMAKE_SOURCE_DIR}/shaders/model.vert
    ${CMAKE_SOURCE_DIR}/shaders/model.frag
)

set(SPIRV_OUTPUTS "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSLC} ${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV})
endforeach()

add_custom_target(Shaders DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(ForceUnleashedVR Shaders)

target_compile_definitions(ForceUnleashedVR PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}")
