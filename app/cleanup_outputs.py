
"""Automatically clean up old ImageTo3D reconstructions.

Each completed reconstruction is stored as:

outputs/api/<UUID>/
    depth_map.png
    model.ply

Examples:

Dry run:
    python -m app.cleanup_outputs

Delete expired reconstructions:
    python -m app.cleanup_outputs --apply

Custom retention period:
    python -m app.cleanup_outputs --retention-hours 72 --apply
"""

import argparse
import shutil
import time

from pathlib import Path
from uuid import UUID


# -----------------------------------------
# Configuration
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api"

DEFAULT_RETENTION_HOURS = 48


# -----------------------------------------
# Validate reconstruction directories
# -----------------------------------------

def is_reconstruction_directory(directory: Path) -> bool:
    """Return True if directory is a UUID-named folder."""

    if directory.is_symlink() or not directory.is_dir():
        return False

    try:
        identifier = UUID(directory.name)

    except (ValueError, AttributeError):
        return False

    return str(identifier) == directory.name


# -----------------------------------------
# Determine completion timestamp
# -----------------------------------------

def get_completion_timestamp(directory: Path):
    """Return the latest modification time of a completed reconstruction."""

    depth_path = directory / "depth_map.png"
    mesh_path = directory / "model.ply"

    # Do not follow unexpected symlinks.
    if depth_path.is_symlink() or mesh_path.is_symlink():
        return None

    # Both files must exist.
    if not depth_path.is_file() or not mesh_path.is_file():
        return None

    try:
        return max(
            depth_path.stat().st_mtime,
            mesh_path.stat().st_mtime,
            directory.stat().st_mtime,
        )

    except OSError:
        return None


# -----------------------------------------
# Cleanup implementation
# -----------------------------------------

def cleanup_reconstructions(
    retention_hours: float = DEFAULT_RETENTION_HOURS,
    apply: bool = False,
):
    """Find and optionally delete expired reconstruction outputs."""

    if not 0 < retention_hours < float("inf"):
        raise ValueError("Retention hours must be positive and finite.")

    if not OUTPUT_ROOT.is_dir():
        print(f"Output directory does not exist: {OUTPUT_ROOT}")
        return 0

    cutoff_timestamp = (
        time.time() - retention_hours * 60 * 60
    )

    eligible = 0
    deleted = 0
    errors = 0

    print("=" * 55)
    print("ImageTo3D Reconstruction Cleanup")
    print("=" * 55)

    print(f"Output directory : {OUTPUT_ROOT}")
    print(f"Retention period : {retention_hours:g} hours")
    print(f"Execution mode   : {'DELETE' if apply else 'DRY RUN'}")

    print("-" * 55)

    for directory in sorted(OUTPUT_ROOT.iterdir()):

        # Skip directories that are not reconstruction IDs.
        if not is_reconstruction_directory(directory):
            continue

        # Skip incomplete reconstructions.
        completed_at = get_completion_timestamp(directory)

        if completed_at is None:
            continue

        # Skip reconstructions within the retention period.
        if completed_at >= cutoff_timestamp:
            continue

        eligible += 1

        if not apply:
            print(f"[DRY RUN] Would delete: {directory.name}")
            continue

        try:
            shutil.rmtree(directory)

            deleted += 1

            print(f"[DELETED] {directory.name}")

        except OSError as exc:
            errors += 1

            print(f"[ERROR] {directory.name}: {exc}")

    print("-" * 55)

    print(f"Eligible : {eligible}")
    print(f"Deleted  : {deleted}")
    print(f"Errors   : {errors}")

    print("=" * 55)

    if errors:
        raise RuntimeError(
            f"Cleanup encountered {errors} deletion error(s)."
        )

    return deleted


# -----------------------------------------
# Command-line interface
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean up expired ImageTo3D reconstruction outputs."
    )

    parser.add_argument(
        "--retention-hours",
        type=float,
        default=DEFAULT_RETENTION_HOURS,
        help="Number of hours to retain completed reconstructions.",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete expired reconstruction directories.",
    )

    args = parser.parse_args()

    cleanup_reconstructions(
        retention_hours=args.retention_hours,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
