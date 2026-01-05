# Delete only demo_1
# ./isaaclab.sh -p scripts/tools/delete_episodes.py -i ./datasets/so_arm_demos3.hdf5 -o ./datasets/cleaned.hdf5 -d 1

# Delete demo_1 and demo_3
# ./isaaclab.sh -p scripts/tools/delete_episodes.py -i ./datasets/so_arm_demos4.hdf5 -o ./datasets/cleaned.hdf5 -d 1 3


import h5py
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Delete specific episodes from HDF5 dataset by index.")
    parser.add_argument("-i", "--input_file", required=True, help="Input HDF5 file path")
    parser.add_argument("-o", "--output_file", required=True, help="Output HDF5 file path")
    # Receive only numbers here
    parser.add_argument("-d", "--delete_nums", nargs="+", required=True, help="Episode numbers to delete (e.g., 1 3 5)")

    args = parser.parse_args()

    # Automatically prefix with 'demo_'
    delete_ids = [f"demo_{n}" for n in args.delete_nums]

    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        return

    with h5py.File(args.input_file, "r") as src, h5py.File(args.output_file, "w") as dst:
        # Create 'data' group and copy attributes (crucial for Isaac Lab env_args)
        data_group = dst.create_group("data")
        if "data" in src:
            for attr_key, attr_val in src["data"].attrs.items():
                data_group.attrs[attr_key] = attr_val

        # Sort episodes numerically to ensure correct re-indexing
        sorted_keys = sorted(src["data"].keys(), key=lambda x: int(x.split('_')[1]))
        
        new_idx = 0
        for ep_id in sorted_keys:
            if ep_id in delete_ids:
                print(f"❌ Removing: {ep_id}")
                continue
            
            # Copy and rename to keep the sequence continuous (0, 1, 2...)
            src.copy(f"data/{ep_id}", dst, f"data/demo_{new_idx}")
            print(f"✅ Keeping {ep_id} -> New ID: demo_{new_idx}")
            new_idx += 1

        # Copy other metadata like 'mask' if they exist
        for key in src.keys():
            if key != "data":
                src.copy(key, dst)

    print(f"\n✨ Done! Deleted numbers: {args.delete_nums}")
    print(f"Total episodes remaining: {new_idx}")

if __name__ == "__main__":
    main()