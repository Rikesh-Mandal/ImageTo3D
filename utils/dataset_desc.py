import h5py

h5_path = r'E:\Dissertation\Trial\ImageTo3D\ImageTo3D\data\NYU_Depth_V2\train\basement_0001a\00006.h5'

# Open the HDF5 file in read-only mode
with h5py.File(h5_path, 'r') as f:
    # List the names of all datasets and groups in the file
    print("Keys in the HDF5 file:", list(f.keys()))

    # see the structure of a specific dataset or group
    for key in f.keys():
        print(f"Dataset/Group: {key}")
        print(f"Shape: {f[key].shape}")
        print(f"Datatype: {f[key].dtype}")
        print(f"Description: {f[key]}")
        print('-' * 50)
