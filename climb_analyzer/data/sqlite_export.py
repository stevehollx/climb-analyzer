#!/usr/bin/env python3
"""
SQLite Export Module

Generates SQLite databases from climb analysis data for iOS app compatibility.
Uses iOS-compatible schema with camelCase columns, UUID primary keys, geohash
columns, R-tree spatial index, and file_stats table.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)


def compute_geohash(latitude: float, longitude: float, precision: int) -> str:
    """
    Compute geohash for given coordinates.

    MUST match iOS implementation exactly.
    Base32 alphabet (note: no 'a', 'i', 'l', 'o'):
    "0123456789bcdefghjkmnpqrstuvwxyz"

    Args:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        precision: Number of characters (1-12, use 6 for most cases)

    Returns:
        Geohash string of specified precision
    """
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    hash_str = ""
    is_even = True  # Start with longitude
    bit = 0
    bits = 0

    while len(hash_str) < precision:
        if is_even:
            # Longitude
            mid = (lon_range[0] + lon_range[1]) / 2
            if longitude >= mid:  # >= per geohash spec
                bit |= (1 << (4 - bits))
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            # Latitude
            mid = (lat_range[0] + lat_range[1]) / 2
            if latitude >= mid:  # >= per geohash spec
                bit |= (1 << (4 - bits))
                lat_range[0] = mid
            else:
                lat_range[1] = mid

        is_even = not is_even
        bits += 1

        if bits == 5:
            hash_str += base32[bit]
            bits = 0
            bit = 0

    return hash_str


def test_geohash():
    """Test geohash implementation matches standard algorithm (pygeohash verified)."""
    test_cases = [
        (37.3414, -121.6426, 6, "9q9krv"),   # Mount Hamilton, CA
        (37.3894, -122.2589, 6, "9q9h9y"),   # Old La Honda Road, CA
        (47.6062, -122.3321, 6, "c23nb6"),   # Seattle, WA
        (19.8069, -155.6310, 6, "8e92dt"),   # Hawaii
        # Additional precision tests
        (37.3414, -121.6426, 1, "9"),        # Single char precision
        (37.3414, -121.6426, 3, "9q9"),      # 3 char precision
    ]

    for lat, lon, precision, expected in test_cases:
        result = compute_geohash(lat, lon, precision)
        assert result == expected, f"Geohash mismatch at ({lat}, {lon}): got {result}, expected {expected}"

    print("All geohash tests passed!")
    return True


# iOS-compatible SQLite schema with camelCase columns
CLIMBS_TABLE_SCHEMA = """
CREATE TABLE climbs (
    id TEXT PRIMARY KEY,
    fileId TEXT NOT NULL,
    streetName TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    distanceFromCenter REAL,
    lat REAL,
    lon REAL,
    cyclingAccess TEXT,
    category TEXT,
    basicScore REAL,
    fietsScore REAL,
    pdiScore REAL,
    elevationGain REAL,
    height REAL,
    prominence REAL,
    length REAL,
    avgGrade REAL,
    maxGrade REAL,
    highwayType TEXT,
    surface TEXT,
    tracktype TEXT,
    wayId TEXT,
    osmLink TEXT,
    allWayIds TEXT,
    connectedClimbs TEXT,
    elevationProfile TEXT,
    elevationProfileUnits TEXT,
    geohash_p1 TEXT,
    geohash_p2 TEXT,
    geohash_p3 TEXT,
    geohash_p4 TEXT,
    geohash_p5 TEXT,
    geohash_p6 TEXT
)
"""

FILE_STATS_TABLE_SCHEMA = """
CREATE TABLE file_stats (
    fileId TEXT PRIMARY KEY,
    climbCount INTEGER,
    maxLength REAL,
    maxProminence REAL,
    maxElevGain REAL,
    maxAvgGrade REAL,
    maxMaxGrade REAL,
    maxHeight REAL,
    maxBasicScore REAL,
    maxFietsScore REAL,
    maxPdiScore REAL,
    importDate TEXT
)
"""

RTREE_SCHEMA = """
CREATE VIRTUAL TABLE climbs_rtree USING rtree(
    id,
    minLat, maxLat,
    minLon, maxLon
)
"""

RTREE_MAP_SCHEMA = """
CREATE TABLE climb_rtree_map (
    rtree_id INTEGER PRIMARY KEY,
    climb_id TEXT NOT NULL
)
"""

# All required indexes (18+)
CLIMBS_INDEXES = [
    # Single column indexes
    "CREATE INDEX idx_climbs_fileId ON climbs(fileId)",
    "CREATE INDEX idx_climbs_lat ON climbs(lat)",
    "CREATE INDEX idx_climbs_lon ON climbs(lon)",
    "CREATE INDEX idx_climbs_category ON climbs(category)",
    "CREATE INDEX idx_climbs_basicScore ON climbs(basicScore)",
    "CREATE INDEX idx_climbs_fietsScore ON climbs(fietsScore)",
    "CREATE INDEX idx_climbs_pdiScore ON climbs(pdiScore)",
    "CREATE INDEX idx_climbs_elevationGain ON climbs(elevationGain)",
    "CREATE INDEX idx_climbs_prominence ON climbs(prominence)",
    "CREATE INDEX idx_climbs_length ON climbs(length)",
    "CREATE INDEX idx_climbs_avgGrade ON climbs(avgGrade)",
    # Composite spatial index
    "CREATE INDEX idx_climbs_location ON climbs(lat, lon)",
    # R-tree mapping index
    "CREATE INDEX idx_rtree_map_climb ON climb_rtree_map(climb_id)",
    # Geohash compound indexes
    "CREATE INDEX idx_climbs_geohash_p1_file ON climbs(geohash_p1, fileId)",
    "CREATE INDEX idx_climbs_geohash_p2_file ON climbs(geohash_p2, fileId)",
    "CREATE INDEX idx_climbs_geohash_p3_file ON climbs(geohash_p3, fileId)",
    "CREATE INDEX idx_climbs_geohash_p4_file ON climbs(geohash_p4, fileId)",
    "CREATE INDEX idx_climbs_geohash_p5_file ON climbs(geohash_p5, fileId)",
    "CREATE INDEX idx_climbs_geohash_p6_file ON climbs(geohash_p6, fileId)",
]

# Column mapping from Excel headers to iOS SQLite columns
# Excel columns have units in parentheses which we need to strip
EXCEL_TO_SQLITE_COLUMNS = {
    "Street Name": "streetName",
    "City": "city",
    "State": "state",
    "Country": "country",
    "Latitude": "lat",
    "Longitude": "lon",
    "Cycling": "cyclingAccess",
    "Category": "category",
    "Basic Score": "basicScore",
    "FIETS Score": "fietsScore",
    "PDI Score": "pdiScore",
    "Highway Type": "highwayType",
    "Surface": "surface",
    "Tracktype": "tracktype",
    "Start Way ID": "wayId",
    "OSM Link": "osmLink",
    "All Way IDs": "allWayIds",
    "Connected Climbs": "connectedClimbs",
}

# Columns with unit suffixes that need pattern matching
UNIT_COLUMNS = {
    "From Center": "distanceFromCenter",
    "Elev Gain": "elevationGain",
    "Height": "height",
    "Prominence": "prominence",
    "Length": "length",
    "Elevation Profile": "elevationProfile",
}

# Fixed columns (always same name)
GRADE_COLUMNS = {
    "Avg Grade (%)": "avgGrade",
    "Max Grade (%)": "maxGrade",
}


class SQLiteExporter:
    """Exports climb data to SQLite database with iOS-compatible schema."""

    def __init__(self, output_path: Path, file_id: str, units: str = "ft", batch_size: int = 5000):
        """
        Initialize SQLite exporter.

        Args:
            output_path: Path for the SQLite database file
            file_id: Identifier for this file (e.g., "hawaii.db")
            units: "ft" for imperial, "m" for metric
            batch_size: Number of rows to insert per batch (default 5000)
        """
        self.output_path = Path(output_path)
        self.file_id = file_id
        self.units = units
        self.batch_size = batch_size
        self.conn: Optional[sqlite3.Connection] = None
        self.row_count = 0
        self._column_mapping: Optional[Dict[str, str]] = None

    def __enter__(self):
        """Context manager entry - open database."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close database."""
        self.close()
        return False

    def open(self):
        """Open database connection and create schema."""
        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing database if present
        if self.output_path.exists():
            self.output_path.unlink()

        self.conn = sqlite3.connect(str(self.output_path))

        # Configure for fast writes
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=MEMORY")

        # Create schema
        self.conn.execute(CLIMBS_TABLE_SCHEMA)
        self.conn.execute(FILE_STATS_TABLE_SCHEMA)
        self.conn.execute(RTREE_SCHEMA)
        self.conn.execute(RTREE_MAP_SCHEMA)
        self.conn.commit()

        self.row_count = 0
        logger.info(f"Created SQLite database: {self.output_path}")

    def close(self):
        """Close database connection, build R-tree, create indexes, and vacuum."""
        if self.conn:
            # Build R-tree from climbs
            self.conn.execute("""
                INSERT INTO climbs_rtree (id, minLat, maxLat, minLon, maxLon)
                SELECT rowid, lat, lat, lon, lon FROM climbs WHERE lat IS NOT NULL AND lon IS NOT NULL
            """)

            # Build R-tree mapping
            self.conn.execute("""
                INSERT INTO climb_rtree_map (rtree_id, climb_id)
                SELECT rowid, id FROM climbs
            """)

            # Calculate and insert file stats
            self.conn.execute("""
                INSERT INTO file_stats
                SELECT
                    ? as fileId,
                    COUNT(*) as climbCount,
                    MAX(length) as maxLength,
                    MAX(prominence) as maxProminence,
                    MAX(elevationGain) as maxElevGain,
                    MAX(avgGrade) as maxAvgGrade,
                    MAX(maxGrade) as maxMaxGrade,
                    MAX(height) as maxHeight,
                    MAX(basicScore) as maxBasicScore,
                    MAX(fietsScore) as maxFietsScore,
                    MAX(pdiScore) as maxPdiScore,
                    ? as importDate
                FROM climbs
            """, (self.file_id, datetime.now().isoformat()))

            self.conn.commit()

            # Create indexes after all data is inserted (faster)
            for index_sql in CLIMBS_INDEXES:
                self.conn.execute(index_sql)
            self.conn.commit()

            # Switch to normal mode and vacuum
            self.conn.execute("PRAGMA journal_mode=DELETE")
            self.conn.execute("VACUUM")
            self.conn.close()
            self.conn = None

            logger.info(f"SQLite database closed with {self.row_count} rows")

    def _build_column_mapping(self, excel_columns: List[str]) -> Dict[str, str]:
        """
        Build mapping from Excel column names to SQLite column names.

        Args:
            excel_columns: List of Excel column names (may include units)

        Returns:
            Dictionary mapping Excel column names to SQLite column names
        """
        mapping = {}

        for excel_col in excel_columns:
            # Check exact matches first
            if excel_col in EXCEL_TO_SQLITE_COLUMNS:
                mapping[excel_col] = EXCEL_TO_SQLITE_COLUMNS[excel_col]
            elif excel_col in GRADE_COLUMNS:
                mapping[excel_col] = GRADE_COLUMNS[excel_col]
            else:
                # Check for unit suffix columns (e.g., "From Center (km)")
                for prefix, sqlite_col in UNIT_COLUMNS.items():
                    if excel_col.startswith(prefix):
                        mapping[excel_col] = sqlite_col
                        break

        return mapping

    def write_rows(self, rows: List[Dict]) -> int:
        """
        Write rows to database.

        Args:
            rows: List of dictionaries with Excel-style column names

        Returns:
            Number of rows written
        """
        if not rows:
            return 0

        if not self.conn:
            raise RuntimeError("Database not open. Call open() first.")

        # Build column mapping on first batch
        if self._column_mapping is None:
            self._column_mapping = self._build_column_mapping(list(rows[0].keys()))

        # All iOS columns in order
        ios_columns = [
            "id", "fileId", "streetName", "city", "state", "country",
            "distanceFromCenter", "lat", "lon", "cyclingAccess", "category",
            "basicScore", "fietsScore", "pdiScore", "elevationGain", "height",
            "prominence", "length", "avgGrade", "maxGrade", "highwayType",
            "surface", "tracktype", "wayId", "osmLink", "allWayIds",
            "connectedClimbs", "elevationProfile", "elevationProfileUnits",
            "geohash_p1", "geohash_p2", "geohash_p3", "geohash_p4", "geohash_p5", "geohash_p6"
        ]

        placeholders = ", ".join(["?" for _ in ios_columns])
        columns_str = ", ".join(ios_columns)
        insert_sql = f"INSERT INTO climbs ({columns_str}) VALUES ({placeholders})"

        # Convert rows to tuples with iOS schema
        values = []
        for row in rows:
            # Extract values using column mapping
            mapped_values = {}
            for excel_col, sqlite_col in self._column_mapping.items():
                value = row.get(excel_col)
                if value == "" or value == "None":
                    value = None
                mapped_values[sqlite_col] = value

            # Get lat/lon for geohash calculation
            lat = mapped_values.get("lat")
            lon = mapped_values.get("lon")

            # Compute geohashes if coordinates are available
            geohashes = {}
            if lat is not None and lon is not None:
                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                    for precision in range(1, 7):
                        geohashes[f"p{precision}"] = compute_geohash(lat_f, lon_f, precision)
                except (ValueError, TypeError):
                    pass

            # Build row tuple in iOS column order
            row_values = (
                str(uuid.uuid4()),  # id - generate UUID
                self.file_id,  # fileId
                mapped_values.get("streetName"),
                mapped_values.get("city"),
                mapped_values.get("state"),
                mapped_values.get("country"),
                mapped_values.get("distanceFromCenter"),
                lat,
                lon,
                mapped_values.get("cyclingAccess"),
                mapped_values.get("category"),
                float(mapped_values.get("basicScore") or 0) if mapped_values.get("basicScore") is not None else None,
                mapped_values.get("fietsScore"),
                mapped_values.get("pdiScore"),
                mapped_values.get("elevationGain"),
                mapped_values.get("height"),
                mapped_values.get("prominence"),
                mapped_values.get("length"),
                mapped_values.get("avgGrade"),
                mapped_values.get("maxGrade"),
                mapped_values.get("highwayType"),
                mapped_values.get("surface"),
                mapped_values.get("tracktype"),
                mapped_values.get("wayId"),
                mapped_values.get("osmLink"),
                mapped_values.get("allWayIds"),
                mapped_values.get("connectedClimbs"),
                mapped_values.get("elevationProfile"),
                self.units,  # elevationProfileUnits
                geohashes.get("p1"),
                geohashes.get("p2"),
                geohashes.get("p3"),
                geohashes.get("p4"),
                geohashes.get("p5"),
                geohashes.get("p6"),
            )
            values.append(row_values)

        # Batch insert
        self.conn.executemany(insert_sql, values)
        self.conn.commit()

        self.row_count += len(rows)
        return len(rows)

    def get_row_count(self) -> int:
        """Get total rows written."""
        return self.row_count


def save_climbs_to_sqlite(
    rows_iterator: Iterator[Dict],
    output_path: Path,
    file_id: str,
    units: str = "ft",
    total_rows: Optional[int] = None,
    batch_size: int = 5000,
    progress_callback: Optional[callable] = None,
) -> Tuple[Path, int]:
    """
    Stream climb data to SQLite database.

    Args:
        rows_iterator: Iterator yielding dictionaries with Excel-style column names
        output_path: Path for the SQLite database file
        file_id: Identifier for this file (e.g., "hawaii.db")
        units: "ft" for imperial, "m" for metric
        total_rows: Optional total row count for progress reporting
        batch_size: Number of rows to insert per batch
        progress_callback: Optional callback(rows_written, total_rows) for progress

    Returns:
        Tuple of (output_path, row_count)
    """
    with SQLiteExporter(output_path, file_id=file_id, units=units, batch_size=batch_size) as exporter:
        batch = []

        for row in rows_iterator:
            batch.append(row)

            if len(batch) >= batch_size:
                exporter.write_rows(batch)
                if progress_callback:
                    progress_callback(exporter.row_count, total_rows)
                batch = []

        # Write final batch
        if batch:
            exporter.write_rows(batch)
            if progress_callback:
                progress_callback(exporter.row_count, total_rows)

        return output_path, exporter.row_count


def verify_ios_database(db_path: str) -> bool:
    """
    Verify database matches iOS app expectations.

    Args:
        db_path: Path to SQLite database

    Returns:
        True if valid, False otherwise
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    errors = []

    # Check tables exist
    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    required_tables = {'climbs', 'file_stats', 'climbs_rtree', 'climb_rtree_map'}
    missing_tables = required_tables - table_names
    if missing_tables:
        errors.append(f"Missing tables: {missing_tables}")

    # Check climbs columns
    columns = cursor.execute("PRAGMA table_info(climbs)").fetchall()
    column_names = {c[1] for c in columns}
    required_cols = {
        'id', 'fileId', 'streetName', 'city', 'state', 'country',
        'distanceFromCenter', 'lat', 'lon', 'cyclingAccess', 'category',
        'basicScore', 'fietsScore', 'pdiScore', 'elevationGain', 'height',
        'prominence', 'length', 'avgGrade', 'maxGrade', 'highwayType',
        'surface', 'tracktype', 'wayId', 'osmLink', 'allWayIds',
        'connectedClimbs', 'elevationProfile', 'elevationProfileUnits',
        'geohash_p1', 'geohash_p2', 'geohash_p3', 'geohash_p4', 'geohash_p5', 'geohash_p6'
    }
    missing_cols = required_cols - column_names
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check id is TEXT (not INTEGER)
    id_col = next((c for c in columns if c[1] == 'id'), None)
    if id_col and id_col[2].upper() != 'TEXT':
        errors.append(f"Column 'id' should be TEXT, got {id_col[2]}")

    # Check indexes exist (at least 18)
    indexes = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    ).fetchall()
    if len(indexes) < 18:
        errors.append(f"Expected at least 18 indexes, got {len(indexes)}")

    # Check R-tree has data matching climbs
    rtree_count = cursor.execute("SELECT COUNT(*) FROM climbs_rtree").fetchone()[0]
    climb_count = cursor.execute("SELECT COUNT(*) FROM climbs WHERE lat IS NOT NULL").fetchone()[0]
    if rtree_count != climb_count:
        errors.append(f"R-tree count ({rtree_count}) != climb count ({climb_count})")

    # Check file_stats has data
    stats_count = cursor.execute("SELECT COUNT(*) FROM file_stats").fetchone()[0]
    if stats_count == 0:
        errors.append("file_stats table is empty")

    # Check geohashes are populated
    null_geohash = cursor.execute(
        "SELECT COUNT(*) FROM climbs WHERE lat IS NOT NULL AND geohash_p6 IS NULL"
    ).fetchone()[0]
    if null_geohash > 0:
        errors.append(f"{null_geohash} climbs have NULL geohash despite having coordinates")

    # Check UUIDs are valid format
    sample_ids = cursor.execute("SELECT id FROM climbs LIMIT 5").fetchall()
    for (id_val,) in sample_ids:
        try:
            uuid.UUID(id_val)
        except ValueError:
            errors.append(f"Invalid UUID format: {id_val}")
            break

    conn.close()

    if errors:
        print("Verification FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"Database verified: {climb_count} climbs, iOS-compatible")
        return True


def convert_xlsx_to_sqlite(
    xlsx_path: Path,
    sqlite_path: Optional[Path] = None,
    file_id: Optional[str] = None,
    units: str = "ft",
    batch_size: int = 5000,
) -> Tuple[Path, int]:
    """
    Convert an existing XLSX file to SQLite database.

    Args:
        xlsx_path: Path to source Excel file
        sqlite_path: Path for output SQLite file (default: same name with .sqlite extension)
        file_id: Identifier for this file (default: filename without extension)
        units: "ft" for imperial, "m" for metric
        batch_size: Number of rows to process per batch

    Returns:
        Tuple of (sqlite_path, row_count)
    """
    import pandas as pd

    if sqlite_path is None:
        sqlite_path = xlsx_path.with_suffix(".sqlite")

    if file_id is None:
        file_id = xlsx_path.stem + ".db"

    logger.info(f"Converting {xlsx_path} to SQLite...")

    # Read Excel in chunks for memory efficiency
    with SQLiteExporter(sqlite_path, file_id=file_id, units=units, batch_size=batch_size) as exporter:
        # Read with openpyxl for large files
        for chunk in pd.read_excel(xlsx_path, engine="openpyxl", chunksize=batch_size):
            rows = chunk.to_dict(orient="records")
            exporter.write_rows(rows)
            logger.debug(f"Wrote {exporter.row_count} rows...")

        return sqlite_path, exporter.row_count


def export_climbs_with_partitioning(
    climbs: List[Dict],
    output_dir: Path,
    base_filename: str,
    region_key: str,
    file_id: str,
    units: str = "ft",
    batch_size: int = 5000,
) -> Tuple[List[Path], List]:
    """
    Export climbs to SQLite with automatic partitioning if needed.

    For regions exceeding 1.5GB estimated size, this function will:
    1. Try predefined Geofabrik partitions (e.g., NorCal/SoCal for California)
    2. Fall back to Quadtree subdivision if no predefined partitions exist

    Args:
        climbs: List of climb dictionaries with lat/lon keys
        output_dir: Directory for output files
        base_filename: Base name for output files (without extension)
        region_key: Region identifier for partition lookup
        file_id: File identifier for SQLite metadata
        units: "ft" for imperial, "m" for metric
        batch_size: Number of rows per batch insert

    Returns:
        Tuple of (list of SQLite file paths, list of PartitionInfo objects)
        If no partitioning needed, returns ([single_file], [])
    """
    from climb_analyzer.data.partition_engine import (
        needs_partitioning,
        partition_climbs,
        get_climbs_for_partition,
        PartitionInfo,
    )
    from climb_analyzer.data.geo_definitions import get_predefined_partitions

    # Check if partitioning is needed
    if not needs_partitioning(len(climbs)):
        # Single file export
        sqlite_path = output_dir / f"{base_filename}.sqlite"
        with SQLiteExporter(sqlite_path, file_id=file_id, units=units, batch_size=batch_size) as exporter:
            # Process in batches
            for i in range(0, len(climbs), batch_size):
                batch = climbs[i:i + batch_size]
                exporter.write_rows(batch)

        logger.info(f"Exported {len(climbs):,} climbs to single file: {sqlite_path}")
        return [sqlite_path], []

    # Get partition definitions
    predefined = get_predefined_partitions(region_key)
    partitions = partition_climbs(climbs, predefined_partitions=predefined)

    logger.info(f"Partitioning {len(climbs):,} climbs into {len(partitions)} partitions")

    sqlite_files = []
    partition_infos = []

    for partition in partitions:
        # Get climbs for this partition
        partition_climbs_list = get_climbs_for_partition(climbs, partition)

        if not partition_climbs_list:
            logger.warning(f"Partition {partition.partition_id} has no climbs after filtering")
            continue

        # Create partition-specific filename
        partition_filename = f"{base_filename}_{partition.partition_id}.sqlite"
        sqlite_path = output_dir / partition_filename
        partition_file_id = f"{file_id}_{partition.partition_id}"

        # Export partition
        with SQLiteExporter(sqlite_path, file_id=partition_file_id, units=units, batch_size=batch_size) as exporter:
            for i in range(0, len(partition_climbs_list), batch_size):
                batch = partition_climbs_list[i:i + batch_size]
                exporter.write_rows(batch)

        # Update partition info with file details
        partition.file_path = sqlite_path
        partition.file_size = sqlite_path.stat().st_size
        partition.climb_count = len(partition_climbs_list)

        sqlite_files.append(sqlite_path)
        partition_infos.append(partition)

        size_mb = partition.file_size / (1024 * 1024)
        logger.info(f"  {partition.display_name}: {partition.climb_count:,} climbs ({size_mb:.1f} MB)")

    # Generate partition metadata files and checksums
    if partition_infos:
        logger.info("Generating partition metadata files...")
        checksums_content = []

        for partition in partition_infos:
            # Write individual metadata file
            metadata_path = write_partition_metadata(partition, output_dir)
            logger.info(f"  {metadata_path.name}")

            # Collect checksum for combined file
            sha256 = compute_sha256(partition.file_path)
            checksums_content.append(f"{sha256}  {partition.file_path.name}")

        # Write combined checksums file
        checksums_file = output_dir / f"{base_filename}.partitions.sha256"
        with open(checksums_file, "w") as f:
            f.write("\n".join(checksums_content) + "\n")
        logger.info(f"Created partition checksums: {checksums_file.name}")

    return sqlite_files, partition_infos


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hex string of the SHA256 hash
    """
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def write_partition_metadata(partition, output_dir: Path) -> Path:
    """
    Write partition metadata JSON file alongside SQLite.

    Creates a .metadata.json file containing:
    - partition_id, display_name
    - bounds (min_lat, min_lon, max_lat, max_lon)
    - climb_count, file_size_bytes, file_size_mb
    - sha256 checksum
    - created_at timestamp

    Args:
        partition: PartitionInfo object with partition details
        output_dir: Directory for the metadata file

    Returns:
        Path to the created metadata file
    """
    from datetime import datetime

    metadata = {
        "partition_id": partition.partition_id,
        "display_name": partition.display_name,
        "bounds": {
            "min_lat": partition.bounds[0],
            "min_lon": partition.bounds[1],
            "max_lat": partition.bounds[2],
            "max_lon": partition.bounds[3],
        } if partition.bounds else None,
        "climb_count": partition.climb_count,
        "file_size_bytes": partition.file_size,
        "file_size_mb": round(partition.file_size / (1024**2), 1) if partition.file_size else None,
        "sha256": compute_sha256(partition.file_path) if partition.file_path else None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    metadata_path = partition.file_path.parent / f"{partition.file_path.name}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


if __name__ == "__main__":
    # Run geohash tests when module is executed directly
    test_geohash()
