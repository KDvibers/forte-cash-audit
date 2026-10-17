"""Create a neutral, synthetic MOV for development; no bank footage is bundled."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import imageio_ffmpeg


def create_sample(path: Path, *, force: bool = False) -> Path:
    path = path.resolve()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
        '-y' if force else '-n', '-f', 'lavfi', '-i',
        'testsrc2=size=320x240:rate=10:duration=12', '-an', '-c:v', 'libx264',
        '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-f', 'mov', str(path),
    ]
    subprocess.run(command, check=True, timeout=30, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / '.local' / 'sample.mov')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    print(create_sample(args.output, force=args.force))
