#!/usr/bin/env python3
"""
Phase 1: CO3D Frame Sampling

Extracts frames from CO3D video sequences and saves:
- Resized RGB images (max edge 512, 16-pixel aligned)
- Foreground masks
- Sampling metadata

Usage:
    python scripts/phase1_sampling.py --co3d-root data/co3d --output output

Output structure:
    output/phase1_sampled/{category}_{object}_{seq}/
    ├── 5_views/
    │   ├── images/
    │   ├── masks/
    │   └── sampling_info.json
    ├── 10_views/
    └── 20_views/
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from config import SamplingConfig, PathConfig
from core.co3d import CO3DSequence, find_sequences, get_sequence_id
from utils.image import resize_image, resize_mask
from utils.timer import PipelineTimer


def sample_sequence(
    sequence: CO3DSequence,
    output_dir: Path,
    n_views: int,
    config: SamplingConfig,
) -> dict:
    """
    Sample frames from a CO3D sequence.

    Args:
        sequence: CO3DSequence object
        output_dir: Output directory for this view count
        output_dir: Output directory
        n_views: Number of views to sample
        config: Sampling configuration

    Returns:
        Sampling metadata dict
    """
    # Get frame indices
    frame_indices = sequence.get_frame_indices(n_views, strategy=config.strategy)

    # Create output directories
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frames, extracted_indices = sequence.extract_frames(
        frame_indices=frame_indices
    )

    # Extract masks
    try:
        masks = sequence.extract_masks(frame_indices=np.array(extracted_indices))
    except FileNotFoundError:
        print(f"    Warning: No mask video, creating default masks")
        masks = [np.ones(f.shape[:2], dtype=np.uint8) for f in frames]

    # Process and save
    saved_images = []
    original_size = None
    processed_size = None

    for i, (frame, mask, idx) in enumerate(zip(frames, masks, extracted_indices)):
        # Resize image
        resized_frame, info = resize_image(
            frame,
            max_edge=config.max_edge,
            multiple_of=config.multiple_of,
        )

        if original_size is None:
            original_size = info['original_size']
            processed_size = info['new_size']

        # Resize mask to match
        resized_mask = resize_mask(mask, processed_size)

        # Save
        filename = f"frame_{idx:06d}.png"
        img_path = images_dir / filename
        mask_path = masks_dir / filename

        # Convert RGB to BGR for OpenCV
        cv2.imwrite(str(img_path), cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), resized_mask * 255)

        saved_images.append(filename)

    # Save metadata
    metadata = {
        'sequence_path': str(sequence.seq_path),
        'category': sequence.category,
        'object_name': sequence.object_name,
        'sequence_id': sequence.sequence_id,
        'n_views': n_views,
        'strategy': config.strategy,
        'frame_indices': [int(i) for i in extracted_indices],
        'frame_files': saved_images,
        'original_size': list(original_size) if original_size else None,
        'processed_size': list(processed_size) if processed_size else None,
        'max_edge': config.max_edge,
        'multiple_of': config.multiple_of,
        'total_video_frames': sequence.total_frames,
        'fps': sequence.fps,
    }

    with open(output_dir / 'sampling_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Phase 1: CO3D Frame Sampling')
    parser.add_argument('--co3d-root', type=str, default='data/co3d',
                        help='Path to CO3D dataset')
    parser.add_argument('--output', type=str, default='output',
                        help='Output root directory')
    parser.add_argument('--n-views', type=int, nargs='+', default=[5, 10, 20],
                        help='Number of views to sample')
    parser.add_argument('--max-edge', type=int, default=512,
                        help='Maximum image edge length')
    parser.add_argument('--strategy', type=str, default='uniform_time',
                        choices=['uniform_time', 'uniform', 'random'],
                        help='Sampling strategy')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Process specific sequence only')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum sequences to process')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Filter by categories')

    args = parser.parse_args()

    # Setup paths
    co3d_root = Path(args.co3d_root)
    output_root = Path(args.output) / 'phase1_sampled'

    if not co3d_root.exists():
        print(f"Error: CO3D root not found: {co3d_root}")
        return

    # Config
    config = SamplingConfig(
        n_views=args.n_views,
        strategy=args.strategy,
        max_edge=args.max_edge,
    )

    # Find sequences
    if args.sequence:
        seq_path = co3d_root / args.sequence
        if not seq_path.exists():
            print(f"Error: Sequence not found: {seq_path}")
            return
        sequences = [seq_path]
    else:
        sequences = find_sequences(
            co3d_root,
            categories=args.categories,
            max_sequences=args.max_sequences,
        )

    print(f"Found {len(sequences)} sequences")
    print(f"Sampling {config.n_views} views each")
    print(f"Output: {output_root}")
    print()

    # Initialize Timer
    timer = PipelineTimer(total_steps=len(sequences), name="Phase1")

    # Process sequences
    for i, seq_path in enumerate(sequences):
        seq_id = get_sequence_id(seq_path, co3d_root)
        print(f"[{i+1}/{len(sequences)}] {seq_id}")

        try:
            sequence = CO3DSequence(seq_path)
            print(f"  Frames: {sequence.total_frames}, FPS: {sequence.fps:.1f}")

            for n_views in config.n_views:
                output_dir = output_root / seq_id / f"{n_views}_views"
                print(f"  Sampling {n_views} views -> {output_dir}")

                metadata = sample_sequence(sequence, output_dir, n_views, config)
                print(f"    Saved {len(metadata['frame_files'])} frames")
                print(f"    Size: {metadata['original_size']} -> {metadata['processed_size']}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

        print(f"  {timer.step()}")
        print()

    timer.save_stats(str(Path(args.output) / "phase1_stats.json"))
    print("Phase 1 complete!")


if __name__ == '__main__':
    main()
