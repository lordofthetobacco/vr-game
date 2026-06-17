#include "include/io.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

char *read_binary_file(const char *filename, size_t *size) {
    printf("Loading file: %s\n", filename);
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(file_size);
    if (!buffer) {
        fprintf(stderr, "Could not allocate memory for file: %s\n", filename);
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, file_size, file);
    fclose(file);

    if (size) {
        *size = file_size;
    }

    return buffer;
}