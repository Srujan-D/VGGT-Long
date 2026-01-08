"""
Helper script to save metadata for JEPA patch learning.

This extracts chunk_indices and loop_list from VGGT-Long's output
and saves them to metadata.json.

Usage:
    python jepa_patch/save_metadata_helper.py ./output --num_images 5000 --chunk_size 60 --overlap 30

Or if you have loop_closures.txt:
    python jepa_patch/save_metadata_helper.py ./output --auto
"""

import argparse
import json
from pathlib import Path
import numpy as np
import glob


def compute_chunk_indices(num_images, chunk_size, overlap):
    """
    Compute chunk indices given sequence length and chunking parameters.
    """
    if num_images <= chunk_size:
        return [(0, num_images)]

    step = chunk_size - overlap
    num_chunks = (num_images - overlap + step - 1) // step

    chunk_indices = []
    for i in range(num_chunks):
        start_idx = i * step
        end_idx = min(start_idx + chunk_size, num_images)
        chunk_indices.append((start_idx, end_idx))

    return chunk_indices


def load_loop_closures(output_dir):
    """
    Try to load loop closures from various possible locations.
    """
    loop_list = []

    # Try loop_closures.txt (created by SALAD detector)
    loop_file = output_dir / 'loop_closures.txt'
    if loop_file.exists():
        print(f"Found loop closures at {loop_file}")
        with open(loop_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    i, j = int(parts[0]), int(parts[1])
                    loop_list.append((i, j))

    return loop_list


def infer_from_chunks(output_dir):
    """
    Infer metadata from saved chunk files.
    """
    unaligned_dir = output_dir / '_tmp_results_unaligned'

    if not unaligned_dir.exists():
        raise FileNotFoundError(f"Cannot find {unaligned_dir}. Did VGGT-Long run successfully?")

    # Find all chunk files
    chunk_files = sorted(glob.glob(str(unaligned_dir / 'chunk_*.npy')))

    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No chunk files found in {unaligned_dir}")

    print(f"Found {len(chunk_files)} chunk files")

    # Load first chunk to get dimensions
    chunk_0 = np.load(chunk_files[0], allow_pickle=True).item()
    chunk_size = chunk_0['depth'].shape[0]

    print(f"Detected chunk size: {chunk_size}")

    # Infer overlap from chunk structure
    if len(chunk_files) > 1:
        # Load second chunk
        chunk_1 = np.load(chunk_files[1], allow_pickle=True).item()

        # Chunk 0: frames [0, chunk_size)
        # Chunk 1: frames [start, start + chunk_size)
        # Overlap = chunk_size - (chunk_1_start - 0)

        # We need to infer this... typical values are 30 for overlap
        # For now, assume standard 50% overlap
        overlap = chunk_size // 2
        print(f"Assuming overlap: {overlap} (50% of chunk size)")
    else:
        overlap = 0
        print("Single chunk, no overlap")

    # Compute total number of images
    if len(chunk_files) == 1:
        num_images = chunk_size
    else:
        step = chunk_size - overlap
        # Last chunk might be partial
        last_chunk = np.load(chunk_files[-1], allow_pickle=True).item()
        last_chunk_frames = last_chunk['depth'].shape[0]

        # num_images = (num_chunks - 1) * step + last_chunk_frames
        num_images = (len(chunk_files) - 1) * step + last_chunk_frames

    print(f"Inferred total images: {num_images}")

    # Compute chunk indices
    chunk_indices = compute_chunk_indices(num_images, chunk_size, overlap)

    return {
        'num_images': num_images,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'chunk_indices': chunk_indices
    }


def main():
    parser = argparse.ArgumentParser(
        description='Save metadata for JEPA patch learning'
    )
    parser.add_argument('output_dir', type=str,
                        help='VGGT-Long output directory')
    parser.add_argument('--auto', action='store_true',
                        help='Automatically infer parameters from chunk files')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Total number of images')
    parser.add_argument('--chunk_size', type=int, default=60,
                        help='Chunk size (default: 60)')
    parser.add_argument('--overlap', type=int, default=30,
                        help='Overlap size (default: 30)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist!")
        return

    print(f"Processing VGGT-Long output from: {output_dir}")

    # Get chunk structure
    if args.auto:
        print("\nAuto-detecting parameters from chunk files...")
        info = infer_from_chunks(output_dir)
        num_images = info['num_images']
        chunk_size = info['chunk_size']
        overlap = info['overlap']
        chunk_indices = info['chunk_indices']
    else:
        if args.num_images is None:
            print("Error: --num_images is required when not using --auto")
            return

        num_images = args.num_images
        chunk_size = args.chunk_size
        overlap = args.overlap

        print(f"\nUsing provided parameters:")
        print(f"  Total images: {num_images}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Overlap: {overlap}")

        chunk_indices = compute_chunk_indices(num_images, chunk_size, overlap)

    print(f"\nComputed {len(chunk_indices)} chunks:")
    for i, (start, end) in enumerate(chunk_indices):
        print(f"  Chunk {i}: frames [{start}, {end})")

    # Load loop closures
    print("\nLoading loop closures...")
    loop_list = load_loop_closures(output_dir)

    if len(loop_list) > 0:
        print(f"Found {len(loop_list)} loop closures")
        # Show first few
        for i, (a, b) in enumerate(loop_list[:5]):
            print(f"  Loop {i}: ({a}, {b})")
        if len(loop_list) > 5:
            print(f"  ... and {len(loop_list) - 5} more")
    else:
        print("No loop closures found (this is OK if loop detection was disabled)")

    # Create metadata
    metadata = {
        'chunk_indices': chunk_indices,
        'loop_list': loop_list,
        'num_images': num_images,
        'chunk_size': chunk_size,
        'overlap': overlap
    }

    # Save
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata to {metadata_path}")
    print("\nYou can now train the patch encoder:")
    print(f"  python jepa_patch/train.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
