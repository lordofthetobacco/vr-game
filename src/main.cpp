#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <glm/glm.hpp>

#include "include/camera.hpp"
#include "include/model.hpp"
#include "include/object.hpp"
#include "include/renderer.hpp"

namespace {
bool endsWith(const char *s, const char *suffix) {
    size_t ls = std::strlen(s), lf = std::strlen(suffix);
    return ls >= lf && std::strcmp(s + ls - lf, suffix) == 0;
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <model.obj|.fbx|.glb|.gltf | object.toml>\n",
                argv[0]);
        return 1;
    }

    // A .toml describes a model plus the four PBR maps; any other path is a raw
    // model rendered with default (matte) textures.
    std::string modelPath = argv[1];
    TextureSet textures;
    if (endsWith(argv[1], ".toml")) {
        ObjectDef def;
        if (!loadObjectToml(argv[1], def)) {
            return 1;
        }
        modelPath = def.modelPath;
        textures.base = def.base;
        textures.normal = def.normal;
        textures.roughness = def.roughness;
        textures.metallic = def.metallic;
    }

    Model model;
    if (!loadModel(modelPath.c_str(), model)) {
        return 1;
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
        return 1;
    }

    SDL_Window *window =
        SDL_CreateWindow("ForceUnleashedVR - Model Viewer", 1280, 720,
                         SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        SDL_Log("Could not create window: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    OrbitCamera camera;
    camera.frame(model.center, model.radius);

    // Light travelling downward (from above) with a slight angle so the shading
    // reads as 3D rather than flat-on-top.
    const glm::vec3 lightDir = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.3f));

    try {
        Renderer renderer(window);
        renderer.uploadObject(model, textures);

        bool quit = false;
        bool dragging = false;
        SDL_Event e;

        while (!quit) {
            while (SDL_PollEvent(&e)) {
                switch (e.type) {
                case SDL_EVENT_QUIT:
                    quit = true;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (e.key.key == SDLK_ESCAPE) {
                        quit = true;
                    }
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    renderer.onResize();
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    if (e.button.button == SDL_BUTTON_LEFT) {
                        dragging = true;
                    }
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    if (e.button.button == SDL_BUTTON_LEFT) {
                        dragging = false;
                    }
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    if (dragging) {
                        camera.rotate(e.motion.xrel, e.motion.yrel);
                    }
                    break;
                case SDL_EVENT_MOUSE_WHEEL:
                    camera.zoom(e.wheel.y);
                    break;
                default:
                    break;
                }
            }

            int w = 0, h = 0;
            SDL_GetWindowSizeInPixels(window, &w, &h);
            float aspect =
                h > 0 ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;

            renderer.drawFrame(camera.viewMatrix(), camera.projMatrix(aspect),
                               lightDir, camera.position());
        }
    } catch (const std::exception &ex) {
        SDL_Log("Fatal: %s", ex.what());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
