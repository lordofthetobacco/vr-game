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

find_package(OpenXR REQUIRED)
if(OpenXR_FOUND)
    message(STATUS "Found OpenXR: ${OpenXR_VERSION}")
else()
    message(FATAL_ERROR "OpenXR not found")
endif()

# --- Dear ImGui (fetched from GitHub) ---
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.8
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)
FetchContent_GetProperties(imgui SOURCE_DIR imgui_SOURCE_DIR)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC SDL3::SDL3 Vulkan::Vulkan)
target_compile_options(imgui_lib PRIVATE -w)

# --- meshoptimizer (fetched from GitHub) ---
FetchContent_Declare(
    meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG        v0.22
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(meshoptimizer)

# --- Bullet3 physics (fetched from GitHub) ---
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(BUILD_BULLET3            OFF CACHE BOOL "" FORCE)
set(BUILD_PYBULLET           OFF CACHE BOOL "" FORCE)
set(BUILD_OPENGL3_DEMOS      OFF CACHE BOOL "" FORCE)
set(BUILD_CPU_DEMOS          OFF CACHE BOOL "" FORCE)
set(BUILD_EXTRAS             OFF CACHE BOOL "" FORCE)
set(BUILD_UNIT_TESTS         OFF CACHE BOOL "" FORCE)
set(BUILD_BULLET2_DEMOS      OFF CACHE BOOL "" FORCE)
set(INSTALL_LIBS             OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(
    bullet3
    GIT_REPOSITORY https://github.com/bulletphysics/bullet3.git
    GIT_TAG        3.25
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(bullet3)

add_executable(ForceUnleashedVR
    src/main.cpp
    src/io.cpp
    src/model.cpp
    src/object.cpp
    src/scene.cpp
    src/physics_world.cpp
    src/renderer.cpp
    src/xr.cpp
    src/stb_image_impl.cpp
)
target_compile_options(ForceUnleashedVR PRIVATE
    "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-Wpedantic>"
)
target_include_directories(ForceUnleashedVR PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(ForceUnleashedVR PRIVATE
    ${bullet3_SOURCE_DIR}/src
)
target_link_libraries(ForceUnleashedVR PRIVATE
    SDL3::SDL3 Vulkan::Vulkan assimp::assimp imgui_lib meshoptimizer
    OpenXR::openxr_loader OpenXR::headers
    BulletDynamics BulletCollision LinearMath
)

# stb_image is third-party single-header code: suppress its warnings.
set_source_files_properties(src/stb_image_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")

# Bullet3 generates many warnings under our strict flags — silence them.
set_source_files_properties(src/physics_world.cpp PROPERTIES COMPILE_OPTIONS "-w")
set_source_files_properties(src/main.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")

# OpenXR structs are initialized with the idiomatic `{XR_TYPE_*}` form (only the
# .type member listed); the rest are intentionally left zero. Silence the
# resulting missing-field-initializer noise for this file only.
set_source_files_properties(src/xr.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-missing-field-initializers")

# --- Shader compilation (GLSL -> SPIR-V) ---
# Fetch and build the reference glslang compiler so no local Vulkan SDK install
# is required. ENABLE_OPT=OFF keeps it self-contained (no SPIRV-Tools dependency).
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG        15.3.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glslang)

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
        # -V: emit a Vulkan SPIR-V binary (stage inferred from the extension).
        COMMAND glslang-standalone -V ${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER} glslang-standalone
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV})
endforeach()

add_custom_target(Shaders DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(ForceUnleashedVR Shaders)

target_compile_definitions(ForceUnleashedVR PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}")
