#include "PNanoVDB_GLSL.h"

pnanovdb_buf_t discarded = { 0 };
pnanovdb_uint32_t gridType;
pnanovdb_readaccessor_t accessor;
pnanovdb_grid_handle_t gridHandle;

void initNanoVolume() {
    pnanovdb_address_t address;
    address.byte_offset = 0;
    gridHandle.address = address;

    pnanovdb_tree_handle_t treeHandle = pnanovdb_grid_get_tree(discarded, gridHandle);
    pnanovdb_root_handle_t rootHandle = pnanovdb_tree_get_root(discarded, treeHandle);
    pnanovdb_readaccessor_init(accessor, rootHandle);
    gridType = pnanovdb_grid_get_grid_type(discarded, gridHandle);
}
