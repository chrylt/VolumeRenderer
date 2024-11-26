// NanoVDB constants
const uint NANO_VDB_MAGIC_NUMBER = 0x3044564E; // 'N' 'V' 'D' '0'

// Node levels
const int ROOT_LEVEL = 0;
const int INTERNAL_LEVEL = 1;
const int LEAF_LEVEL = 2;

// Node dimensions
const uint NODE1_LOG2DIM = 3; // Internal node dimensions (8x8x8)
const uint NODE1_DIM = 1 << NODE1_LOG2DIM; // 8
const uint NODE2_LOG2DIM = 3; // Leaf node dimensions (8x8x8)
const uint NODE2_DIM = 1 << NODE2_LOG2DIM; // 8

// Node counts
const uint NODE1_NUM_VALUES = NODE1_DIM * NODE1_DIM * NODE1_DIM; // 512
const uint NODE2_NUM_VALUES = NODE2_DIM * NODE2_DIM * NODE2_DIM; // 512

// Buffer reference to allow pointer-like access to buffer data
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer NanoVDBBuffer {
    uint data[];
};

// NanoVDB Grid Data Header
struct NanoVDBGridData {
    uint magic;                // Magic number
    uint version;              // Version number
    uint gridType;             // Grid type (e.g., float grid)
    uint gridClass;            // Grid class (e.g., level set, fog volume)
    float worldBBoxMin[3];     // World-space bounding box minimum
    float voxelSize;           // Voxel size
    float worldBBoxMax[3];     // World-space bounding box maximum
    float dummy0;              // Padding to align to 16 bytes
    uint64_t blindDataOffset;  // Offset to blind data (if any)
    uint64_t gridNameOffset;   // Offset to grid name string
    uint64_t gridDataOffset;   // Offset to grid data (Root node)
    uint64_t treeDataOffset;   // Offset to tree data
    // ... Other members as needed
};

// Node structures
struct RootNode {
    uint64_t numInternalNodes;     // Number of internal nodes
    uint64_t numLeafNodes;         // Number of leaf nodes
    // Child nodes or tiles follow
};

struct InternalNode {
    uint offset;               // Offset to child nodes or values
    uint16_t childMask[64];        // Child mask for the 8x8x8 voxels
    // Child pointers or values follow
};

struct LeafNode {
    uint8_t valueMask[512];        // Value mask for the 8x8x8 voxels
    float values[512];             // Voxel values
};

// NanoVDB Sampler
struct NanoVDBSampler {
    NanoVDBBuffer vdbBuffer;      // Buffer reference to the NanoVDB grid data
    NanoVDBGridData gridData;  // Grid metadata
};

// Helper function to count set bits before a given bit index in child masks
uint countSetBitsBefore(uint16_t childMask[64], uint bitIdx) {
    uint count = 0;
    uint totalBits = bitIdx;
    for (uint i = 0; i < totalBits / 16; ++i) {
        count += bitCount(childMask[i]);
    }
    uint16_t mask = childMask[totalBits / 16];
    for (uint i = 0; i < totalBits % 16; ++i) {
        if ((mask & (1 << i)) != 0) {
            count++;
        }
    }
    return count;
}

// Helper function to get the size of child node at a given level
uint sizeOfChildNodeAtLevel(int level) {
    if (level == INTERNAL_LEVEL) {
        return uint(sizeof(InternalNode));
    }
    else if (level == LEAF_LEVEL) {
        return uint(sizeof(LeafNode));
    }
    return 0;
}


// Function to initialize the NanoVDB vdbSampler
void nanoVdbInit(inout NanoVDBSampler vdbSampler, uint64_t bufferAddress) {
    vdbSampler.vdbBuffer = NanoVDBBuffer(bufferAddress);

    // Read the grid data header
    vdbSampler.gridData.magic = vdbSampler.vdbBuffer.data[0];
    vdbSampler.gridData.version = vdbSampler.vdbBuffer.data[1];
    vdbSampler.gridData.gridType = vdbSampler.vdbBuffer.data[2];
    vdbSampler.gridData.gridClass = vdbSampler.vdbBuffer.data[3];

    // Read world bounding box minimum
    vdbSampler.gridData.worldBBoxMin[0] = uintBitsToFloat(vdbSampler.vdbBuffer.data[4]);
    vdbSampler.gridData.worldBBoxMin[1] = uintBitsToFloat(vdbSampler.vdbBuffer.data[5]);
    vdbSampler.gridData.worldBBoxMin[2] = uintBitsToFloat(vdbSampler.vdbBuffer.data[6]);

    // Read voxel size
    vdbSampler.gridData.voxelSize = uintBitsToFloat(vdbSampler.vdbBuffer.data[7]);

    // Read world bounding box maximum
    vdbSampler.gridData.worldBBoxMax[0] = uintBitsToFloat(vdbSampler.vdbBuffer.data[8]);
    vdbSampler.gridData.worldBBoxMax[1] = uintBitsToFloat(vdbSampler.vdbBuffer.data[9]);
    vdbSampler.gridData.worldBBoxMax[2] = uintBitsToFloat(vdbSampler.vdbBuffer.data[10]);

    // Read offsets
    vdbSampler.gridData.blindDataOffset = uint64_t(vdbSampler.vdbBuffer.data[11]) |
        (uint64_t(vdbSampler.vdbBuffer.data[12]) << 32);
    vdbSampler.gridData.gridNameOffset = uint64_t(vdbSampler.vdbBuffer.data[13]) |
        (uint64_t(vdbSampler.vdbBuffer.data[14]) << 32);
    vdbSampler.gridData.gridDataOffset = uint64_t(vdbSampler.vdbBuffer.data[15]) |
        (uint64_t(vdbSampler.vdbBuffer.data[16]) << 32);
    vdbSampler.gridData.treeDataOffset = uint64_t(vdbSampler.vdbBuffer.data[17]) |
        (uint64_t(vdbSampler.vdbBuffer.data[18]) << 32);

    // Verify magic number
    if (vdbSampler.gridData.magic != NANO_VDB_MAGIC_NUMBER) {
        // Handle error (e.g., set a flag, return a default value)
    }
}

// Helper function to read data from the buffer
uint readUInt(NanoVDBBuffer vdbBuffer, uint64_t offset) {
    return vdbBuffer.data[uint(offset / 4)];    // attention! maybe this casts leads to bugs further down the line! (other casts like this in shader also exist
}

uint64_t readUInt64(NanoVDBBuffer vdbBuffer, uint64_t offset) {
    uint low = vdbBuffer.data[uint(offset / 4)];
    uint high = vdbBuffer.data[uint(offset / 4 + 1)];
    return uint64_t(low) | (uint64_t(high) << 32);
}

float readFloat(NanoVDBBuffer vdbBuffer, uint64_t offset) {
    uint bits = readUInt(vdbBuffer, offset);
    return uintBitsToFloat(bits);
}

// Function to sample the NanoVDB grid at a given world-space position
float nanoVdbSample(in NanoVDBSampler vdbSampler, vec3 position) {
    // Convert world-space position to index-space position
    vec3 indexPos = (position - vec3(vdbSampler.gridData.worldBBoxMin[0],
        vdbSampler.gridData.worldBBoxMin[1],
        vdbSampler.gridData.worldBBoxMin[2])) / vdbSampler.gridData.voxelSize;

    // Start traversal from the root node
    uint64_t nodeOffset = vdbSampler.gridData.gridDataOffset;
    int level = ROOT_LEVEL;

    while (true) {
        if (level == ROOT_LEVEL) {
            // Root node
            uint64_t rootNodeAddr = nodeOffset;

            // Read root node
            RootNode rootNode;
            rootNode.numInternalNodes = readUInt64(vdbSampler.vdbBuffer, rootNodeAddr);
            rootNode.numLeafNodes = readUInt64(vdbSampler.vdbBuffer, rootNodeAddr + 8);

            // For simplicity, assume the root node directly points to the first internal node
            nodeOffset = rootNodeAddr + 16; // Skip the RootNode header
            level = INTERNAL_LEVEL;

        }
        else if (level == INTERNAL_LEVEL) {
            // Internal node
            uint64_t internalNodeAddr = nodeOffset;

            // Read internal node data
            InternalNode internalNode;
            internalNode.offset = readUInt(vdbSampler.vdbBuffer, internalNodeAddr);

            for (int i = 0; i < 64; ++i) {
                internalNode.childMask[i] = uint16_t(readUInt(vdbSampler.vdbBuffer, internalNodeAddr + 4 + i * 2));
            }

            // Compute child index based on indexPos
            uint ix = uint(indexPos.x) >> NODE1_LOG2DIM;
            uint iy = uint(indexPos.y) >> NODE1_LOG2DIM;
            uint iz = uint(indexPos.z) >> NODE1_LOG2DIM;

            uint childIdx = (ix & 0x7) << 6 | (iy & 0x7) << 3 | (iz & 0x7);

            // Check if the child exists
            uint16_t mask = internalNode.childMask[childIdx / 16];
            uint16_t bit = uint16_t(1 << (childIdx % 16));
            if ((mask & bit) == 0) {
                // Child is inactive, return background value
                return 0.0;
            }

            // Child is active, proceed to the child node
            uint childCount = countSetBitsBefore(internalNode.childMask, childIdx);
            nodeOffset = internalNodeAddr + internalNode.offset + childCount * sizeOfChildNodeAtLevel(level + 1);
            level = LEAF_LEVEL;

        }
        else if (level == LEAF_LEVEL) {
            // Leaf node
            uint64_t leafNodeAddr = nodeOffset;

            // Read leaf node data
            LeafNode leafNode;
            for (int i = 0; i < 512; ++i) {
                leafNode.valueMask[i] = uint8_t(readUInt(vdbSampler.vdbBuffer, leafNodeAddr + i));
            }

            for (int i = 0; i < 512; ++i) {
                leafNode.values[i] = readFloat(vdbSampler.vdbBuffer, leafNodeAddr + 512 + i * 4);
            }

            // Compute voxel indices within the leaf node
            uint ix = uint(indexPos.x) & (NODE2_DIM - 1);
            uint iy = uint(indexPos.y) & (NODE2_DIM - 1);
            uint iz = uint(indexPos.z) & (NODE2_DIM - 1);

            uint voxelIdx = ix * NODE2_DIM * NODE2_DIM + iy * NODE2_DIM + iz;

            // Check if the voxel is active
            if (leafNode.valueMask[voxelIdx] == 0) {
                // Voxel is inactive, return background value
                return 0.0;
            }

            // Perform trilinear interpolation
            // For simplicity, we'll fetch the value directly
            float value = leafNode.values[voxelIdx];

            return value;
        }
        else {
            // Invalid level
            break;
        }
    }

    // If traversal fails, return background value
    return 0.0;
}
