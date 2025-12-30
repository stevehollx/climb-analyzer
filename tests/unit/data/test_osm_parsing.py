"""
Unit tests for OSM data parsing and spatial indexing.

Tests cover:
- Spatial index loading
- JSONL metadata parsing
- Bounding box queries
- Way metadata extraction
- Index file validation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
from pathlib import Path
import tempfile
import shutil

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from climb_analyzer.data.spatial_index import SpatialIndexManager


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for index files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_index_files(temp_index_dir):
    """Create mock index files."""
    osm_file = temp_index_dir / "test.osm.pbf"
    osm_file.touch()

    cache_dir = temp_index_dir / "osm_indexes"
    cache_dir.mkdir(exist_ok=True)

    # Create mock rtree index files
    idx_file = cache_dir / "test.osm_spatial.idx"
    dat_file = cache_dir / "test.osm_spatial.dat"
    idx_file.touch()
    dat_file.touch()

    # Create mock metadata file
    metadata_file = cache_dir / "test.osm_metadata.jsonl"

    # Write sample metadata
    with open(metadata_file, 'w') as f:
        # Header
        header = {
            "_index_metadata": {
                "total_ways": 3,
                "created_at": "2024-01-01",
                "surface_filter": "all",
                "cycling_only": True,
            }
        }
        f.write(json.dumps(header) + '\n')

        # Way entries
        ways = [
            {"id": 12345, "data": {"name": "Main St", "highway": "primary", "nodes": 10}},
            {"id": 67890, "data": {"name": "Oak Ave", "highway": "residential", "nodes": 5}},
            {"id": 11111, "data": {"name": "Hill Rd", "highway": "tertiary", "nodes": 8}},
        ]
        for way in ways:
            f.write(json.dumps(way) + '\n')

    return {
        "osm_file": str(osm_file),
        "cache_dir": str(cache_dir),
        "metadata_file": metadata_file,
    }


@pytest.fixture
def spatial_index_manager(mock_index_files):
    """Create a SpatialIndexManager instance."""
    return SpatialIndexManager(
        osm_file_path=mock_index_files["osm_file"],
        cache_dir=mock_index_files["cache_dir"],
        silent=True,
    )


@pytest.fixture
def sample_way_metadata():
    """Sample way metadata for testing."""
    return {
        12345: {
            "name": "Main Street",
            "highway": "primary",
            "surface": "asphalt",
            "nodes": 15,
            "length": 2500.0,
        },
        67890: {
            "name": "Oak Avenue",
            "highway": "residential",
            "surface": "asphalt",
            "nodes": 8,
            "length": 1200.0,
        },
    }


# ============================================================================
# Initialization Tests
# ============================================================================

class TestSpatialIndexManagerInit:
    """Test SpatialIndexManager initialization."""

    def test_initialization(self, mock_index_files):
        """Test basic initialization."""
        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
        )

        assert manager.osm_file_path == mock_index_files["osm_file"]
        assert manager.cache_dir == Path(mock_index_files["cache_dir"])
        assert manager.spatial_idx is None
        assert manager.way_metadata == {}

    def test_initialization_with_silent_mode(self, mock_index_files):
        """Test initialization with silent mode."""
        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
            silent=True,
        )

        assert manager.silent is True

    def test_file_paths_calculation(self, mock_index_files):
        """Test that file paths are calculated correctly."""
        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
        )

        # Check index file path
        assert manager.idx_file.name == "test.osm_spatial"
        assert manager.metadata_file.name == "test.osm_metadata.jsonl"


# ============================================================================
# Index Existence Tests
# ============================================================================

class TestIndexExistence:
    """Test index file existence checking."""

    def test_exists_with_all_files(self, spatial_index_manager):
        """Test exists() when all files present."""
        assert spatial_index_manager.exists() is True

    def test_exists_missing_idx_file(self, spatial_index_manager):
        """Test exists() when .idx file is missing."""
        idx_file = Path(str(spatial_index_manager.idx_file) + ".idx")
        idx_file.unlink()

        assert spatial_index_manager.exists() is False

    def test_exists_missing_dat_file(self, spatial_index_manager):
        """Test exists() when .dat file is missing."""
        dat_file = Path(str(spatial_index_manager.idx_file) + ".dat")
        dat_file.unlink()

        assert spatial_index_manager.exists() is False

    def test_exists_missing_metadata_file(self, spatial_index_manager):
        """Test exists() when metadata file is missing."""
        spatial_index_manager.metadata_file.unlink()

        assert spatial_index_manager.exists() is False

    def test_exists_with_nonexistent_directory(self, temp_index_dir):
        """Test exists() when directory doesn't exist."""
        manager = SpatialIndexManager(
            osm_file_path=str(temp_index_dir / "nonexistent.osm.pbf"),
            cache_dir=str(temp_index_dir / "nonexistent_cache"),
        )

        assert manager.exists() is False


# ============================================================================
# Metadata Loading Tests
# ============================================================================

class TestMetadataLoading:
    """Test JSONL metadata file loading."""

    def test_load_metadata(self, spatial_index_manager):
        """Test loading metadata from JSONL file."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            spatial_index_manager.load_index(silent=True)

            # Check that metadata was loaded
            assert len(spatial_index_manager.way_metadata) > 0
            assert "_index_metadata" in spatial_index_manager.way_metadata

    def test_load_metadata_way_entries(self, spatial_index_manager):
        """Test that way entries are loaded correctly."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            spatial_index_manager.load_index(silent=True)

            # Check specific way IDs
            assert 12345 in spatial_index_manager.way_metadata
            assert 67890 in spatial_index_manager.way_metadata
            assert 11111 in spatial_index_manager.way_metadata

            # Check way data
            assert spatial_index_manager.way_metadata[12345]["name"] == "Main St"
            assert spatial_index_manager.way_metadata[67890]["highway"] == "residential"

    def test_load_metadata_header(self, spatial_index_manager):
        """Test that metadata header is loaded correctly."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            spatial_index_manager.load_index(silent=True)

            metadata = spatial_index_manager.way_metadata["_index_metadata"]
            assert metadata["total_ways"] == 3
            assert metadata["surface_filter"] == "all"
            assert metadata["cycling_only"] is True

    def test_load_metadata_empty_file(self, temp_index_dir, mock_index_files):
        """Test loading from empty metadata file."""
        # Create empty metadata file
        metadata_file = Path(mock_index_files["cache_dir"]) / "test.osm_metadata.jsonl"
        metadata_file.write_text("")

        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
        )

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            with pytest.raises(Exception):  # Should raise error on empty file
                manager.load_index(silent=True)

    def test_load_metadata_corrupted_json(self, mock_index_files):
        """Test loading from corrupted JSONL file."""
        # Write corrupted JSON
        metadata_file = Path(mock_index_files["cache_dir"]) / "test.osm_metadata.jsonl"
        metadata_file.write_text("invalid json\n")

        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
        )

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            with pytest.raises(json.JSONDecodeError):
                manager.load_index(silent=True)


# ============================================================================
# Index Loading Tests
# ============================================================================

class TestIndexLoading:
    """Test rtree spatial index loading."""

    def test_load_index_success(self, spatial_index_manager):
        """Test successful index loading."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            mock_rtree.Index.return_value = mock_idx

            spatial_index_manager.load_index(silent=True)

            # Should have loaded the index
            assert spatial_index_manager.spatial_idx is not None
            mock_rtree.Index.assert_called_once()

    def test_load_index_missing_files(self, spatial_index_manager):
        """Test loading when index files don't exist."""
        # Remove index files
        idx_file = Path(str(spatial_index_manager.idx_file) + ".idx")
        idx_file.unlink()

        with pytest.raises(FileNotFoundError):
            spatial_index_manager.load_index(silent=True)

    def test_load_index_missing_metadata(self, spatial_index_manager):
        """Test loading when metadata file is missing."""
        spatial_index_manager.metadata_file.unlink()

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            with pytest.raises(FileNotFoundError):
                spatial_index_manager.load_index(silent=True)

    def test_load_index_rtree_not_available(self, spatial_index_manager):
        """Test loading when rtree library is not available."""
        with patch('climb_analyzer.data.spatial_index.index', side_effect=ImportError):
            with pytest.raises(ImportError):
                spatial_index_manager.load_index(silent=True)

    def test_load_index_verbose_mode(self, spatial_index_manager, capsys):
        """Test that verbose mode prints progress."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            spatial_index_manager.load_index(silent=False)

            captured = capsys.readouterr()
            # Should have printed some output (metadata loading message)
            assert "Loading metadata" in captured.out or "Loaded" in captured.out

    def test_load_index_silent_mode(self, spatial_index_manager, capsys):
        """Test that silent mode suppresses output."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            spatial_index_manager.load_index(silent=True)

            captured = capsys.readouterr()
            # Should not print in silent mode
            assert captured.out == ""


# ============================================================================
# Index Query Tests
# ============================================================================

class TestIndexQueries:
    """Test spatial index querying."""

    def test_query_bbox_basic(self, spatial_index_manager):
        """Test basic bounding box query."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            mock_idx.intersection.return_value = [12345, 67890]
            mock_rtree.Index.return_value = mock_idx

            spatial_index_manager.load_index(silent=True)

            # Query bounding box
            bbox = (35.0, -82.6, 35.1, -82.5)  # (min_lat, min_lon, max_lat, max_lon)
            results = list(spatial_index_manager.spatial_idx.intersection(bbox))

            assert len(results) == 2
            assert 12345 in results
            assert 67890 in results

    def test_query_bbox_empty_result(self, spatial_index_manager):
        """Test bounding box query with no results."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            mock_idx.intersection.return_value = []
            mock_rtree.Index.return_value = mock_idx

            spatial_index_manager.load_index(silent=True)

            bbox = (40.0, -90.0, 41.0, -89.0)  # Far from data
            results = list(spatial_index_manager.spatial_idx.intersection(bbox))

            assert len(results) == 0

    def test_query_bbox_with_metadata_lookup(self, spatial_index_manager):
        """Test querying and retrieving metadata."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            mock_idx.intersection.return_value = [12345]
            mock_rtree.Index.return_value = mock_idx

            spatial_index_manager.load_index(silent=True)

            bbox = (35.0, -82.6, 35.1, -82.5)
            way_ids = list(spatial_index_manager.spatial_idx.intersection(bbox))

            # Retrieve metadata for results
            for way_id in way_ids:
                metadata = spatial_index_manager.way_metadata.get(way_id)
                assert metadata is not None
                assert "name" in metadata


# ============================================================================
# Reload Tests
# ============================================================================

class TestIndexReload:
    """Test index reloading functionality."""

    def test_reload_index(self, spatial_index_manager):
        """Test reloading an already loaded index."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            mock_rtree.Index.return_value = mock_idx

            # Load initially
            spatial_index_manager.load_index(silent=True)
            assert spatial_index_manager.spatial_idx is not None

            # Reload
            spatial_index_manager.reload_index()
            assert spatial_index_manager.spatial_idx is not None

            # Should have called Index twice
            assert mock_rtree.Index.call_count == 2


# ============================================================================
# Create Index Tests
# ============================================================================

class TestIndexCreation:
    """Test index creation (not supported in this module)."""

    def test_create_index_raises_error(self, spatial_index_manager):
        """Test that create_index raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            spatial_index_manager.create_index()


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_way_id_not_found(self, spatial_index_manager):
        """Test retrieving metadata for non-existent way ID."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            spatial_index_manager.load_index(silent=True)

            # Query non-existent way
            metadata = spatial_index_manager.way_metadata.get(99999)
            assert metadata is None

    def test_metadata_with_missing_fields(self, temp_index_dir, mock_index_files):
        """Test loading metadata with missing fields."""
        metadata_file = Path(mock_index_files["cache_dir"]) / "test.osm_metadata.jsonl"

        with open(metadata_file, 'w') as f:
            # Header
            f.write(json.dumps({"_index_metadata": {"total_ways": 1}}) + '\n')
            # Way with minimal data
            f.write(json.dumps({"id": 12345, "data": {"name": "Test"}}) + '\n')

        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
        )

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            manager.load_index(silent=True)

            # Should load successfully with partial data
            assert 12345 in manager.way_metadata
            assert manager.way_metadata[12345]["name"] == "Test"

    def test_large_metadata_file(self, temp_index_dir, mock_index_files):
        """Test loading large metadata file."""
        metadata_file = Path(mock_index_files["cache_dir"]) / "test.osm_metadata.jsonl"

        # Create large metadata file
        with open(metadata_file, 'w') as f:
            # Header
            f.write(json.dumps({"_index_metadata": {"total_ways": 1000}}) + '\n')
            # Many ways
            for i in range(1000):
                f.write(json.dumps({
                    "id": 10000 + i,
                    "data": {"name": f"Street {i}", "highway": "residential"}
                }) + '\n')

        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
        )

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            manager.load_index(silent=True)

            # Should have loaded all ways
            assert len(manager.way_metadata) == 1001  # 1000 ways + metadata header

    def test_unicode_in_metadata(self, temp_index_dir, mock_index_files):
        """Test loading metadata with Unicode characters."""
        metadata_file = Path(mock_index_files["cache_dir"]) / "test.osm_metadata.jsonl"

        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"_index_metadata": {"total_ways": 1}}) + '\n')
            f.write(json.dumps({
                "id": 12345,
                "data": {"name": "Calle José Martí", "highway": "primary"}  # Unicode
            }) + '\n')

        manager = SpatialIndexManager(
            osm_file_path=mock_index_files["osm_file"],
            cache_dir=mock_index_files["cache_dir"],
        )

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            manager.load_index(silent=True)

            # Should handle Unicode correctly
            assert manager.way_metadata[12345]["name"] == "Calle José Martí"


# ============================================================================
# Integration Tests
# ============================================================================

class TestSpatialIndexIntegration:
    """Integration tests for complete workflows."""

    def test_complete_load_and_query_workflow(self, spatial_index_manager):
        """Test complete workflow: load index and query."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            mock_idx.intersection.return_value = [12345, 67890]
            mock_rtree.Index.return_value = mock_idx

            # Check existence
            assert spatial_index_manager.exists() is True

            # Load index
            spatial_index_manager.load_index(silent=True)
            assert spatial_index_manager.spatial_idx is not None
            assert len(spatial_index_manager.way_metadata) > 0

            # Query
            bbox = (35.0, -82.6, 35.1, -82.5)
            way_ids = list(spatial_index_manager.spatial_idx.intersection(bbox))

            # Get metadata
            results = []
            for way_id in way_ids:
                metadata = spatial_index_manager.way_metadata.get(way_id)
                if metadata:
                    results.append((way_id, metadata))

            assert len(results) > 0

    def test_multiple_queries_on_same_index(self, spatial_index_manager):
        """Test multiple queries on the same loaded index."""
        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            mock_idx.intersection.side_effect = [
                [12345],
                [67890],
                [12345, 67890],
            ]
            mock_rtree.Index.return_value = mock_idx

            spatial_index_manager.load_index(silent=True)

            # Multiple queries
            results1 = list(spatial_index_manager.spatial_idx.intersection((35.0, -82.6, 35.05, -82.55)))
            results2 = list(spatial_index_manager.spatial_idx.intersection((35.05, -82.55, 35.1, -82.5)))
            results3 = list(spatial_index_manager.spatial_idx.intersection((35.0, -82.6, 35.1, -82.5)))

            assert len(results1) == 1
            assert len(results2) == 1
            assert len(results3) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
