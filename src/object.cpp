#include "include/object.hpp"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Strip surrounding double or single quotes if present.
std::string unquote(const std::string &s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// Directory portion of a path, including the trailing slash (empty if none).
std::string dirOf(const std::string &path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    return path.substr(0, slash + 1);
}

// Resolve a path read from the toml relative to the toml's directory, unless it
// is already absolute.
std::string resolve(const std::string &dir, const std::string &p) {
    if (p.empty()) return p;
    if (p.front() == '/') return p; // absolute
    return dir + p;
}

} // namespace

bool loadObjectToml(const char *path, ObjectDef &out) {
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "Could not open object file: %s\n", path);
        return false;
    }

    const std::string baseDir = dirOf(path);
    std::string section;
    std::string line;

    while (std::getline(file, line)) {
        // Strip comments (# to end of line).
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);

        line = trim(line);
        if (line.empty()) continue;

        // Section header: [name]
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string value = unquote(trim(line.substr(eq + 1)));
        if (value.empty()) continue;

        if (section.empty() && key == "model") {
            out.modelPath = resolve(baseDir, value);
        } else if (section == "textures") {
            if (key == "base") out.base = resolve(baseDir, value);
            else if (key == "normal") out.normal = resolve(baseDir, value);
            else if (key == "roughness") out.roughness = resolve(baseDir, value);
            else if (key == "metallic") out.metallic = resolve(baseDir, value);
        }
    }

    // All five keys are required.
    struct {
        const char *name;
        const std::string &value;
    } required[] = {
        {"model", out.modelPath},
        {"textures.base", out.base},
        {"textures.normal", out.normal},
        {"textures.roughness", out.roughness},
        {"textures.metallic", out.metallic},
    };

    bool ok = true;
    for (const auto &r : required) {
        if (r.value.empty()) {
            fprintf(stderr, "Object file '%s' is missing required key '%s'\n",
                    path, r.name);
            ok = false;
        }
    }

    return ok;
}
