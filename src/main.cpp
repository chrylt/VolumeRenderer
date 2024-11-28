#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/math/Ray.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <corecrt_math_defines.h>

#include "openvdb/openvdb.h"
#include "nanovdb/util/CreateNanoGrid.h"

// Function to save a PGM image
void savePGM(const std::string& filename, int width, int height, const std::vector<float>& data)
{
    std::ofstream file(filename, std::ios::binary);
    if (file) {
        file << "P5\n" << width << " " << height << "\n255\n";
        for (int i = 0; i < width * height; ++i) {
            unsigned char value = static_cast<unsigned char>(std::min(1.0f, data[i]) * 255.0f);
            file.write(reinterpret_cast<char*>(&value), sizeof(unsigned char));
        }
        file.close();
    }
    else {
        std::cerr << "Failed to save image to " << filename << std::endl;
    }
}

int main(int argc, char** argv)
{
    // Load a NanoVDB grid from file or create a fog volume sphere
    nanovdb::GridHandle<> handle;
    if (argc > 1) {
        handle = nanovdb::io::readGrid<>(argv[1]);
        std::cout << "Loaded NanoVDB grid from " << argv[1] << "\n";
    }
    else {
        handle = nanovdb::tools::createFogVolumeSphere<float>(100.0f, nanovdb::Vec3d(0.0), 1.0, 3.0, nanovdb::Vec3d(0.0), "sphere");
        std::cout << "Created a fog volume sphere grid.\n";
    }

    using GridT = nanovdb::NanoGrid<float>;
    const GridT* grid = handle.grid<float>();
    if (!grid) {
        std::cerr << "Failed to get grid from handle." << std::endl;
        return 1;
    }

    // Load .vdb file and convert to nanovdb
/*
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
        std::cerr << "Error opening file " << filename << ": " << e.what() << std::endl;
        return 1;
    }

    // Read the first FloatGrid from the file
    openvdb::GridBase::Ptr baseGrid;
    openvdb::FloatGrid::Ptr openvdb_grid;

    for (openvdb::io::File::NameIterator nameIter = file.beginName();
        nameIter != file.endName(); ++nameIter) {
        baseGrid = file.readGrid(nameIter.gridName());
        openvdb_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        if (openvdb_grid) {
            std::cout << "Loaded grid: " << nameIter.gridName() << std::endl;
            break;
        }
    }

    file.close();

    if (!openvdb_grid) {
        std::cerr << "No FloatGrid found in " << filename << std::endl;
        openvdb::uninitialize();
        return 1;
    }

    // Convert to NanoVDB Grid
    nanovdb::GridHandle<> handle = nanovdb::tools::createNanoGrid(*openvdb_grid);
    const nanovdb::NanoGrid<float>* grid = handle.grid<float>();*/

    // Image dimensions
    const int width = 512;
    const int height = 512;
    std::vector<float> image(width * height, 0.0f);

    // Camera parameters
    nanovdb::Vec3f camera_pos(0.0f, 0.0f, -300.0f);
    nanovdb::Vec3f look_at(0.0f, 0.0f, 0.0f);
    nanovdb::Vec3f up(0.0f, 1.0f, 0.0f);
    float fov = 45.0f; // Field of view in degrees

    // Precompute camera basis vectors
    nanovdb::Vec3f forward = (look_at - camera_pos).normalize();
    nanovdb::Vec3f right = forward.cross(up).normalize();
    nanovdb::Vec3f up_vec = right.cross(forward);

    float aspect_ratio = float(width) / float(height);
    float scale = tanf(fov * 0.5f * M_PI / 180.0f);

    // Accessor for the grid
    auto accessor = grid->tree().getAccessor();

    // For each pixel
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Compute normalized device coordinates
            float ndc_x = (x + 0.5f) / float(width);
            float ndc_y = (y + 0.5f) / float(height);

            // Screen space coordinates
            float px = (2.0f * ndc_x - 1.0f) * aspect_ratio * scale;
            float py = (1.0f - 2.0f * ndc_y) * scale;

            // Compute ray direction
            nanovdb::Vec3f ray_dir = (forward + px * right + py * up_vec).normalize();

            // Create world-space ray
            nanovdb::Ray<float> wRay(camera_pos, ray_dir);

            // Transform ray to index space
            nanovdb::Ray<float> iRay = wRay.worldToIndexF(*grid);

            // Compute intersection with grid's bounding box
            if (!iRay.clip(grid->worldBBox())) {
                continue; // Ray misses the bounding box
            }

            // Simple raymarching
            float t = iRay.t0();
            float t_end = iRay.t1();
            float dt = 1.0f; // Step size
            float accumulated_density = 0.0f;
            while (t < t_end && accumulated_density < 1.0f) {
                nanovdb::Vec3f pos = iRay(t);
                float density = accessor.getValue(nanovdb::Coord::Floor(pos));
                accumulated_density += density * dt;
                t += dt;
            }

            // Store result in image
            image[y * width + x] = std::min(accumulated_density, 1.0f);
        }
    }

    // Save image to file
    savePGM("output_nanovdb.pgm", width, height, image);

    std::cout << "Rendering completed. Image saved to output.pgm\n";

    return 0;
}
