#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <corecrt_math_defines.h>

#include <glm/common.hpp>
#include <glm/vec3.hpp>
#include <glm/detail/func_geometric.inl>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/math/Ray.h>

#include "openvdb/tools/LevelSetSphere.h"

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

float raymarch(const Ray& ray_nano, nanovdb::NanoGrid<float>* grid) {
    // Simple raymarching function with fixed step size
    const float tMax = 1200.0; // Maximum distance to march
    const float dt = 1.0;     // Step size
    float t = 0.0;            // Current distance along the ray
    float density = 0.0;      // Accumulated density

    while (t < tMax) {

        glm::vec3 position_nano = ray_nano.origin + t * ray_nano.direction;
        glm::ivec3 ijk_nano = glm::floor(position_nano);

        float value = grid->tree().getValue(ijk_nano.x, ijk_nano.y, ijk_nano.z);
        density += value * dt;

        t += dt;
    }
    return density;
}

int main() {
    // Initialize OpenVDB
    openvdb::initialize();

    // Specify the VDB file to read
    const std::string filename = "../../../../resources/bunny_cloud.vdb";

    // Open the VDB file
    openvdb::io::File file(filename);
    try {
        file.open();
    }
    catch (const openvdb::IoError& e) {
        std::cerr << "Error opening file " << filename << ": " << e.what() << '\n';
        return 1;
    }

    // Read the first FloatGrid from the file
    openvdb::GridBase::Ptr baseGrid_openvdb;
    openvdb::FloatGrid::Ptr grid_openvdb;

    for (openvdb::io::File::NameIterator nameIter = file.beginName();
        nameIter != file.endName(); ++nameIter) {
        baseGrid_openvdb = file.readGrid(nameIter.gridName());
        grid_openvdb = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid_openvdb);
        if (grid_openvdb) {
            std::cout << "Loaded grid: " << nameIter.gridName() << '\n';
            break;
        }
    }

    file.close();

    if (!grid_openvdb) {
        std::cerr << "No FloatGrid found in " << filename << '\n';
        openvdb::uninitialize();
        return 1;
    }

    // Create nano test sphere
    nanovdb::GridHandle<> handle = nanovdb::tools::createNanoGrid(*grid_openvdb);
    nanovdb::NanoGrid<float>* grid = handle.grid<float>();


    // Set up the camera parameters
    const int width = 256;
    const int height = 256;
    const glm::vec3 cameraPos(0.0, 250.0, -800.0);
    const float fov = 45.0; // Field of view in degrees

    // Allocate image buffer
    unsigned char* image = new unsigned char[width * height * 3];

    // Precompute camera parameters
    const float aspectRatio = static_cast<float>(width) / height;
    const float scale = tan(fov * 0.5 * M_PI / 180.0);

    // Rendering loop
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Normalize pixel coordinates to [-1, 1]
            float x = (2 * (i + 0.5) / width - 1) * aspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / height) * scale;

            // Compute ray direction
            glm::vec3 rayDir = glm::normalize(glm::vec3(x, y, 1.0));

            Ray ray{ cameraPos, rayDir };

            // Raymarch through the volume
            float density = raymarch(ray, grid);

            // Map density to color (simple grayscale)
            unsigned char color = static_cast<unsigned char>(std::min(density * 5.0, 255.0));

            // Set pixel color
            int index = (j * width + i) * 3;
            image[index + 0] = color; // Red channel
            image[index + 1] = color; // Green channel
            image[index + 2] = color; // Blue channel
        }
    }

    // Save the image to a PPM file
    std::ofstream ofs("output.ppm", std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<char*>(image), width * height * 3);
    ofs.close();

    delete[] image;

    // Clean up OpenVDB
    openvdb::uninitialize();

    std::cout << "Image saved to output.ppm" << '\n';

    return 0;
}
