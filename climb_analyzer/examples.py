"""
Example usage of the climb_analyzer package.

This script demonstrates how to use the various components of the climb_analyzer package.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import climb_analyzer without installation
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_distance_calculation():
    """Example: Calculate distance between two points."""
    from climb_analyzer.utils.helpers import calculate_distance_km

    # Calculate distance from New York City to Los Angeles
    nyc_lat, nyc_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437

    distance = calculate_distance_km(nyc_lat, nyc_lon, la_lat, la_lon)
    print(f"Distance from NYC to LA: {distance:.2f} km")


def example_cycling_access():
    """Example: Determine cycling access from OSM tags."""
    from climb_analyzer.utils.helpers import determine_cycling_access

    # Example 1: Residential road with explicit bicycle tag
    tags1 = {"highway": "residential", "bicycle": "yes"}
    access1 = determine_cycling_access(tags1)
    print(f"Residential road with bicycle=yes: {access1}")

    # Example 2: Motorway (typically no cycling)
    tags2 = {"highway": "motorway"}
    access2 = determine_cycling_access(tags2)
    print(f"Motorway: {access2}")

    # Example 3: Cycleway (designed for cycling)
    tags3 = {"highway": "cycleway"}
    access3 = determine_cycling_access(tags3)
    print(f"Cycleway: {access3}")


def example_climb_dataclasses():
    """Example: Create climb segment and metrics."""
    from climb_analyzer.core.segment import ClimbMetrics, ClimbSegment

    # Create a climb segment
    segment = ClimbSegment(
        name="Alpe d'Huez",
        start_lat=45.0833,
        start_lon=6.0667,
        end_lat=45.0917,
        end_lon=6.0750,
        distance_km=13.8,
        elevation_gain_m=1071,
        avg_gradient=7.9,
        max_gradient=13.0,
        category="HC",
        points=[],  # Would contain (lat, lon, elevation) tuples
    )
    print(f"\nClimb: {segment.name}")
    print(f"Distance: {segment.distance_km} km")
    print(f"Elevation gain: {segment.elevation_gain_m} m")
    print(f"Average gradient: {segment.avg_gradient}%")

    # Create climb metrics
    metrics = ClimbMetrics(
        street_name="D211",
        climb_category="HC",
        climb_score=1150.0,
        elevation_gain=1071.0,
        height=1071.0,
        prominence=1071.0,
        length_km=13.8,
        distance_km=13.8,
        avg_grade=7.9,
        max_grade=13.0,
        min_elevation=720.0,
        max_elevation=1791.0,
        surface="asphalt",
        tracktype="",
        tracktype_definition="",
        way_ids=[123456],
        osm_links=["https://www.openstreetmap.org/way/123456"],
        city_state="Huez, France",
        cycling_access="Yes",
    )
    print(f"\nMetrics for {metrics.street_name}:")
    print(f"Climb score: {metrics.climb_score}")
    print(f"Category: {metrics.climb_category}")


def example_checkpoint_config():
    """Example: Configure checkpoints."""
    from climb_analyzer.processing.checkpoint import CheckpointConfig

    # Create checkpoint configuration
    config = CheckpointConfig(
        time_interval_minutes=5.0,
        progress_milestones=[25, 50, 75, 100],
        save_at_completion=True,
    )

    print("\nCheckpoint Configuration:")
    print(f"Time interval: {config.time_interval_minutes} minutes")
    print(f"Progress milestones: {config.progress_milestones}%")
    print(f"Save at completion: {config.save_at_completion}")


def example_smart_checkpointer():
    """Example: Use smart checkpointer."""
    from climb_analyzer.processing.checkpoint import SmartCheckpointer

    # Create checkpointer for 100 items
    checkpointer = SmartCheckpointer(
        total_items=100, operation_name="Processing Chunks"
    )

    # Simulate processing
    print("\nSimulating checkpoint decisions:")
    for i in [0, 24, 25, 49, 50, 74, 75, 99]:
        should_save = checkpointer.should_checkpoint(i)
        progress = ((i + 1) / 100) * 100
        print(f"Item {i + 1}/100 ({progress:.0f}%): Save checkpoint? {should_save}")


def example_persistence_manager():
    """Example: Create persistence manager."""
    from climb_analyzer.processing.checkpoint import ChunkPersistenceManager

    # Create persistence manager
    persistence = ChunkPersistenceManager(analysis_id="example_analysis")

    print(f"\nPersistence Manager created:")
    print(f"Analysis ID: {persistence.analysis_id}")
    print(f"Base directory: {persistence.base_dir}")
    print(f"Chunk directory: {persistence.chunk_dir}")

    # Note: In real usage, you would save/load data with this manager
    print("\nMethods available:")
    print("- save_chunk()")
    print("- load_chunk()")
    print("- save_progress()")
    print("- load_progress()")
    print("- save_elevation_progress()")
    print("- load_elevation_progress()")


def example_tee():
    """Example: Use Tee class to split output."""
    import sys
    from io import StringIO

    from climb_analyzer.utils.tee import Tee

    # Create a string buffer to capture output
    buffer = StringIO()

    # Create Tee to write to both stdout and buffer
    tee = Tee(sys.stdout, buffer)

    print("\nDemonstrating Tee class:")
    # Write through Tee
    tee.write("This message goes to both stdout and the buffer!\n")
    tee.flush()

    # Show buffer contents
    print("Buffer captured:", repr(buffer.getvalue()))


def example_memory_check():
    """Example: Check and cleanup memory."""
    from climb_analyzer.utils.helpers import check_and_cleanup_memory

    print("\nChecking memory usage:")

    # Check memory without forcing cleanup
    cleaned = check_and_cleanup_memory(threshold_percent=80, force_cleanup=False)
    print(f"Cleanup performed: {cleaned}")

    # Force cleanup
    cleaned = check_and_cleanup_memory(force_cleanup=True)
    print(f"Forced cleanup performed: {cleaned}")


def example_elevation_url():
    """Example: Build elevation API URL."""
    from climb_analyzer.utils.helpers import build_elevation_url

    # Build URLs for different datasets
    datasets = ["srtm30m", "ned10m", "aster30m"]

    print("\nElevation API URLs:")
    for dataset in datasets:
        url = build_elevation_url(dataset)
        if url:
            print(f"{dataset}: {url}")
        else:
            print(f"{dataset}: Not configured (TOPO_API_BASE_URL not set)")


def main():
    """Run all examples."""
    print("=" * 70)
    print("CLIMB ANALYZER PACKAGE EXAMPLES")
    print("=" * 70)

    examples = [
        ("Distance Calculation", example_distance_calculation),
        ("Cycling Access Determination", example_cycling_access),
        ("Climb Data Classes", example_climb_dataclasses),
        ("Checkpoint Configuration", example_checkpoint_config),
        ("Smart Checkpointer", example_smart_checkpointer),
        ("Persistence Manager", example_persistence_manager),
        ("Tee Output Splitting", example_tee),
        ("Memory Check", example_memory_check),
        ("Elevation URL Builder", example_elevation_url),
    ]

    for name, example_func in examples:
        print("\n" + "=" * 70)
        print(f"Example: {name}")
        print("=" * 70)
        try:
            example_func()
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
