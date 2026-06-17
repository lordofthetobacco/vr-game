#ifndef OBJECT_HPP
#define OBJECT_HPP

#include <string>

// An object described by a .toml file: a model plus the four metallic-roughness
// PBR maps. All paths are resolved relative to the .toml file's directory.
struct ObjectDef {
    std::string modelPath;
    std::string base;
    std::string normal;
    std::string roughness;
    std::string metallic;
};

// Parses a .toml object definition. Returns false (with an error printed to
// stderr) if the file cannot be read or any required key is missing.
bool loadObjectToml(const char *path, ObjectDef &out);

#endif
