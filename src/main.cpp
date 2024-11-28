#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <corecrt_math_defines.h>

struct Ray {
    openvdb::Vec3d origin;
    openvdb::Vec3d direction;
};

double raymarch(const Ray& ray, openvdb::FloatGrid::Ptr grid) {
    // Simple raymarching function with fixed step size
    const double tMax = 1200.0; // Maximum distance to march
    const double dt = 1.0;     // Step size
    double t = 0.0;            // Current distance along the ray
    double density = 0.0;      // Accumulated density

    openvdb::FloatGrid::ConstAccessor accessor = grid->getConstAccessor();

    while (t < tMax) {
        openvdb::Vec3d position = ray.origin + t * ray.direction;
        openvdb::Coord ijk = openvdb::Coord::round(position);

        float value = accessor.getValue(ijk);
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
        std::cerr << "Error opening file " << filename << ": " << e.what() << std::endl;
        return 1;
    }

    // Read the first FloatGrid from the file
    openvdb::GridBase::Ptr baseGrid;
    openvdb::FloatGrid::Ptr grid;

    for (openvdb::io::File::NameIterator nameIter = file.beginName();
        nameIter != file.endName(); ++nameIter) {
        baseGrid = file.readGrid(nameIter.gridName());
        grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        if (grid) {
            std::cout << "Loaded grid: " << nameIter.gridName() << std::endl;
            break;
        }
    }

    file.close();

    if (!grid) {
        std::cerr << "No FloatGrid found in " << filename << std::endl;
        openvdb::uninitialize();
        return 1;
    }

    // Set up the camera parameters
    const int width = 256;
    const int height = 256;
    const openvdb::Vec3d cameraPos(0.0, 250.0, -800.0);
    const double fov = 45.0; // Field of view in degrees

    // Allocate image buffer
    unsigned char* image = new unsigned char[width * height * 3];

    // Precompute camera parameters
    const double aspectRatio = static_cast<double>(width) / height;
    const double scale = tan(fov * 0.5 * M_PI / 180.0);

    // Rendering loop
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Normalize pixel coordinates to [-1, 1]
            double x = (2 * (i + 0.5) / width - 1) * aspectRatio * scale;
            double y = (1 - 2 * (j + 0.5) / height) * scale;

            // Compute ray direction
            openvdb::Vec3d rayDir = openvdb::Vec3d(x, y, 1.0).unit();

            Ray ray{ cameraPos, rayDir };

            // Raymarch through the volume
            double density = raymarch(ray, grid);

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

    std::cout << "Image saved to output.ppm" << std::endl;

    return 0;
}
