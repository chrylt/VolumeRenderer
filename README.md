![volume](https://github.com/user-attachments/assets/2a2c03dc-7fb9-4ba0-ba0e-fb09d1b7c239)

# Volume Renderer

This repository provides a real-time volume rendering application that loads an OpenVDB file, converts it to NanoVDB, and visualizes it using several rendering algorithms. The project uses a custom Vulkan abstraction layer (**basalt**), integrates an ImGui-based interface for runtime parameter control, and leverages multiple libraries for handling volume data, window creation, and shading.

## Inspiration and References

This project’s volume rendering algorithms are based on ideas presented in the following papers:

- **[Novák2012Ray]**  
  Jan Novák, Derek Nowrouzezahrai, Carsten Dachsbacher, and Wojciech Jarosz. 2012.  
  *Virtual Ray Lights for Rendering Scenes with Participating Media.*  
  ACM Trans. Graph. 31, 4, Article 60 (July 2012), 11 pages.  
  [https://doi.org/10.1145/2185520.2185556](https://doi.org/10.1145/2185520.2185556)

- **[Novák2012Beam]**  
  Jan Novák, Derek Nowrouzezahrai, Carsten Dachsbacher, and Wojciech Jarosz. 2012.  
  *Progressive Virtual Beam Lights.*  
  Comput. Graph. Forum 31, 4 (June 2012), 1407–1413.  
  [https://doi.org/10.1111/j.1467-8659.2012.03136.x](https://doi.org/10.1111/j.1467-8659.2012.03136.x)

- **[Keller1997]**  
  Alexander Keller. 1997.  
  *Instant radiosity.*  
  In Proceedings of the 24th annual conference on Computer graphics and interactive techniques (SIGGRAPH '97). ACM Press/Addison-Wesley Publishing Co., USA, 49–56.  
  [https://doi.org/10.1145/258734.258769](https://doi.org/10.1145/258734.258769)

- **[Hašan2009]**  
  Miloš Hašan, Jaroslav Křivánek, Bruce Walter, and Kavita Bala. 2009.  
  *Virtual spherical lights for many-light rendering of glossy scenes.*  
  ACM Trans. Graph. 28, 5 (December 2009), 1–6.  
  [https://doi.org/10.1145/1618452.1618489](https://doi.org/10.1145/1618452.1618489)

## Features

- **OpenVDB to NanoVDB Conversion**: Easily load `.vdb` files and convert them to NanoVDB data structures for GPU usage.
- **Multiple Rendering Algorithms**:  
  - Beam [Novák2012Beam]
  - Ray [Novák2012Ray]
  - Point [Keller1997]
  - Sphere [Hašan2009]
  - Path [Keller1997]
- **Configurable Parameters**: Adjust scattering, absorption, camera position, maximum light count, and many other volume rendering parameters on the fly.
- **ImGui Integration**: Real-time tweaking of settings in an overlay GUI.
- **Vulkan Wrapping with basalt**: Simplifies raw Vulkan usage by handling device creation, resource management, synchronization, etc.

## Folder Structure and Resources

The `.vdb` files should reside in a **`resources/` folder** at the **same level** as the `VolumeRenderer` project folder:

```
YourProjectFolder
├─ VolumeRenderer
│  ├─ CMakeLists.txt
│  ├─ src/
│  ├─ shaders/
│  ├─ external/
│  └─ (etc.)
└─ resources
   └─ bunny_cloud.vdb (or other .vdb files)
```

The **compiled shaders** (SPIR-V `.spv` files) are expected to be located **relative to the executable** in `./shaders/compiled_shaders`. The `shaders` subdirectory contains a CMake file that automatically compiles `.vert`, `.frag` or `.comp` sources into `.spv` files during the build.

**Note**: Relative paths for the compiled shaders and resources may vary depending on your setup.

## Building

### Dependencies

Building this project requires:

1. [**Basalt**](https://github.com/...basalt...) (provided as a subdirectory in `external/`)
2. [**OpenVDB**](https://www.openvdb.org/) (subdirectory in `external/`)
3. [**ImGui**](https://github.com/ocornut/imgui) (subdirectory in `external/`)
4. [**glfw3**](https://www.glfw.org/)
5. [**assimp**](https://github.com/assimp/assimp)
6. [**Vulkan SDK**](https://vulkan.lunarg.com/) (for headers, libraries, validation layers)
7. [**CUDAToolkit**](https://developer.nvidia.com/cuda-toolkit) (includes necessary GPU libraries for building parts of NanoVDB / OpenVDB if needed)
8. **TBB** (Threading Building Blocks)

### Build Steps

1. **Clone the repo** and its submodules:

   ```bash
   git clone --recursive https://github.com/chrylt/VolumeRenderer.git
   cd VolumeRenderer
   git submodule update --init --recursive
   ```

2. **Create a build folder** and configure via CMake:

   ```bash
   mkdir build && cd build
   cmake ..
   ```

3. **Compile**:

   ```bash
   cmake --build .
   ```

4. **Run** the resulting executable:

   ```bash
   ./VolumeRenderer
   ```

**Note**: Make sure you have a sibling **`resources/`** folder with your `.vdb` files, as the application by default looks for `../resources/bunny_cloud.vdb` (or whichever file path you specify in the code).

## Usage

1. **Launch** the executable.  
2. The **ImGui** window titled “Settings” lets you:
   - Switch between **Beam**, **Ray**, **Point**, **Sphere**, and **Path** algorithms.
   - Adjust camera, scattering, absorption, step sizes, light source positions, etc.
   - Press **Refresh** to reset the frame accumulation.
3. On **window resize**, the swap chain is automatically recreated, and the rendering is updated to fit the new dimensions.
   
