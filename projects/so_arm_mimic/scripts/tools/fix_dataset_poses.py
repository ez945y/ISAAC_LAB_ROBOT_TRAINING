#!/usr/bin/env python3
"""
Script to fix the target_eef_pose format in HDF5 dataset.

Converts from (T, 7) format [pos(3) + quat(4)] to (T, 4, 4) matrix format.
"""

import argparse
import h5py
import numpy as np
import shutil
from pathlib import Path


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    
    Args:
        quat: Quaternion array of shape (..., 4) in wxyz format
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Build rotation matrix
    rot = np.zeros(quat.shape[:-1] + (3, 3), dtype=quat.dtype)
    
    rot[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rot[..., 0, 1] = 2 * (x * y - w * z)
    rot[..., 0, 2] = 2 * (x * z + w * y)
    
    rot[..., 1, 0] = 2 * (x * y + w * z)
    rot[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rot[..., 1, 2] = 2 * (y * z - w * x)
    
    rot[..., 2, 0] = 2 * (x * z - w * y)
    rot[..., 2, 1] = 2 * (y * z + w * x)
    rot[..., 2, 2] = 1 - 2 * (x * x + y * y)
    
    return rot


def pose_vector_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """Convert pose vector [pos(3) + quat(4)] to 4x4 pose matrix.
    
    Args:
        pose_vec: Pose vector of shape (T, 7) or (7,)
        
    Returns:
        Pose matrix of shape (T, 4, 4) or (4, 4)
    """
    single = pose_vec.ndim == 1
    if single:
        pose_vec = pose_vec[np.newaxis, :]
    
    T = pose_vec.shape[0]
    pos = pose_vec[:, :3]
    quat = pose_vec[:, 3:7]
    
    # Convert quaternion to rotation matrix
    rot = quat_to_rotation_matrix(quat)
    
    # Build 4x4 pose matrix
    pose_mat = np.zeros((T, 4, 4), dtype=pose_vec.dtype)
    pose_mat[:, :3, :3] = rot
    pose_mat[:, :3, 3] = pos
    pose_mat[:, 3, 3] = 1.0
    
    if single:
        pose_mat = pose_mat[0]
    
    return pose_mat


def fix_dataset(input_file: str, output_file: str = None):
    """Fix the target_eef_pose format in the dataset.
    
    Args:
        input_file: Path to input HDF5 file
        output_file: Path to output HDF5 file (default: input_file with _fixed suffix)
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = str(input_path.parent / (input_path.stem + "_fixed" + input_path.suffix))
    
    # Copy file first
    print(f"Copying {input_file} to {output_file}...")
    shutil.copy(input_file, output_file)
    
    # Open and fix
    print(f"Fixing target_eef_pose format...")
    with h5py.File(output_file, 'r+') as f:
        data = f['data']
        
        for demo_key in data.keys():
            demo = data[demo_key]
            
            if 'obs' not in demo or 'datagen_info' not in demo['obs']:
                print(f"  Skipping {demo_key}: no datagen_info")
                continue
            
            dg = demo['obs']['datagen_info']
            
            if 'target_eef_pose' not in dg:
                print(f"  Skipping {demo_key}: no target_eef_pose")
                continue
            
            target_eef_pose = dg['target_eef_pose']
            
            for eef_name in target_eef_pose.keys():
                pose_data = target_eef_pose[eef_name][:]
                
                # Check if already in matrix format
                if pose_data.ndim == 3 and pose_data.shape[-2:] == (4, 4):
                    print(f"  {demo_key}/{eef_name}: Already in matrix format, skipping")
                    continue
                
                # Check if in vector format (T, 7)
                if pose_data.ndim == 2 and pose_data.shape[-1] == 7:
                    print(f"  {demo_key}/{eef_name}: Converting from (T, 7) to (T, 4, 4)")
                    pose_mat = pose_vector_to_matrix(pose_data)
                    
                    # Delete old dataset and create new one
                    del target_eef_pose[eef_name]
                    target_eef_pose.create_dataset(eef_name, data=pose_mat)
                else:
                    print(f"  {demo_key}/{eef_name}: Unknown format {pose_data.shape}, skipping")
    
    print(f"Done! Fixed dataset saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Fix target_eef_pose format in HDF5 dataset")
    parser.add_argument("input_file", type=str, help="Input HDF5 file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output HDF5 file")
    args = parser.parse_args()
    
    fix_dataset(args.input_file, args.output)


if __name__ == "__main__":
    main()
