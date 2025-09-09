#!/usr/bin/env python
# coding: utf-8
# Author: Steve Holl
#
# Version 1.0 - 9-Sep-2025: Initial public release.
#

#Custom libraries
import overpy
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
import requests

#Standard libraries
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Generator
import math
import time
from collections import defaultdict
import sys
import requests
import json
import gc  # For garbage collection
import os
import pickle
from pathlib import Path

from boundaries import country_data, state_data

TOPO_API_URL = "https://api.opentopodata.org/v1/ned10m" #Use this if not setting up a local server
#TOPO_API_URL = 'http://localhost:5000/v1/ned10m'

# Uncomment OVERPASS_API_URL is you are using a local Overpass API server
#OVERPASS_API_URL = "http://localhost:12345/api/interpreter"

# Road surface filtering options
ROAD_SURFACE_FILTERS = {
    'paved': ['highway~"trunk|primary|secondary|tertiary|unclassified|residential|service)"',
              'surface!~"unpaved|gravel|dirt|sand|grass|ground|earth|mud|clay"'],
    'gravel': ['highway~"(track|path|unclassified|tertiary|residential|service"',
               'surface~"gravel|compacted|fine_gravel"'],
    'dirt': ['highway~"track|path"', 'tracktype'],
    'all': ['highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|track"']
}

# Tracktype definitions for display
TRACKTYPE_DEFINITIONS = {
    'grade1': 'Solid: paved or heavily compacted hardcore surface',
    'grade2': 'Mostly solid: unpaved track with through traffic, gravel/dirt with some soft areas',
    'grade3': 'Even mixture of hard and soft materials, gravel/dirt with many soft areas',
    'grade4': 'Mostly soft: soil/sand/grass with some hard material mixed in',
    'grade5': 'Soft: soil/sand/grass, no hard material. Almost impossible for cars'
}

@dataclass
class ClimbSegment:
    """Represents a climbing segment with elevation profile"""
    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    distance_km: float
    elevation_gain_m: float
    avg_gradient: float
    max_gradient: float
    category: str
    points: List[Tuple[float, float, float]]  # (lat, lon, elevation)

@dataclass
class ClimbMetrics:
    street_name: str
    climb_category: str
    climb_score: float
    elevation_gain: float
    height: float
    prominence: float
    length_km: float
    distance_km: float  # Distance between first and last elevation points
    avg_grade: float
    max_grade: float
    min_elevation: float
    max_elevation: float
    surface: str
    tracktype: str  # For tracks, the tracktype classification
    tracktype_definition: str  # Human readable definition
    way_ids: List[int]  # OSM way IDs that make up this road
    osm_links: List[str]  # Links to OSM for each way
    city_state: str = "Unknown"  # City and state information
    distance_from_center_km: float = 0.0  # Distance from search center
    mid_lat: float = 0.0  # Midpoint latitude for location lookup
    mid_lon: float = 0.0  # Midpoint longitude for location lookup
    nodes: List['SimpleNode'] = None  # Nodes for location/distance lookup


class ChunkPersistenceManager:
    """Manages persistent storage of chunk processing progress and data."""
    
    def __init__(self, analysis_id: str):
        """
        Initialize persistence manager with analysis ID.
        
        Args:
            analysis_id: Unique identifier for this analysis session
        """
        self.analysis_id = analysis_id
        self.base_dir = Path("climb_analysis_checkpoints")
        self.analysis_dir = self.base_dir / analysis_id
        self.chunk_dir = self.analysis_dir / "chunks"
        self.elevation_dir = self.analysis_dir / "elevations"  # NEW: elevation checkpoints
        self.progress_file = self.analysis_dir / "progress.pkl"
        self.metadata_file = self.analysis_dir / "metadata.pkl"
        self.elevation_progress_file = self.analysis_dir / "elevation_progress.pkl"  # NEW
        
        # Create directories if they don't exist
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.elevation_dir.mkdir(parents=True, exist_ok=True)  # NEW
    
    def check_for_existing_analysis(self) -> bool:
        """Check if an existing analysis with this ID exists."""
        return self.progress_file.exists() and self.metadata_file.exists()
    
    def get_completion_status(self) -> Tuple[int, int, float]:
        """
        Get completion status of existing analysis.
        
        Returns:
            (completed_chunks, total_chunks, percentage_complete)
        """
        if not self.progress_file.exists():
            return 0, 0, 0.0
        
        try:
            with open(self.progress_file, 'rb') as f:
                progress_data = pickle.load(f)
            
            completed = len(progress_data['processed_chunks'])
            total = progress_data['total_chunks']
            percentage = (completed / total) * 100 if total > 0 else 0.0
            
            return completed, total, percentage
        except Exception as e:
            print(f"Error reading progress file: {e}")
            return 0, 0, 0.0
    
    def load_progress(self) -> Tuple[List[int], int, Dict]:
        """
        Load progress and metadata from disk.
        
        Returns:
            (processed_chunks, total_chunks, metadata)
        """
        processed_chunks = []
        total_chunks = 0
        metadata = {}
        
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                processed_chunks = progress_data.get('processed_chunks', [])
                total_chunks = progress_data.get('total_chunks', 0)
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
        
        except Exception as e:
            print(f"Error loading progress: {e}")
        
        return processed_chunks, total_chunks, metadata
    
    def save_chunk(self, chunk_index: int, chunk_data: List[Dict], chunk_info: Tuple[float, float, float]):
        """
        Save chunk data to disk.
        
        Args:
            chunk_index: Index of the chunk
            chunk_data: List of road segments for this chunk
            chunk_info: (lat, lon, radius) of the chunk
        """
        chunk_file = self.chunk_dir / f"chunk_{chunk_index:04d}.pkl"
        
        try:
            with open(chunk_file, 'wb') as f:
                pickle.dump({
                    'chunk_index': chunk_index,
                    'chunk_data': chunk_data,
                    'chunk_info': chunk_info,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            print(f"Error saving chunk {chunk_index}: {e}")
    
    def load_chunk(self, chunk_index: int) -> Optional[List[Dict]]:
        """
        Load chunk data from disk.
        
        Args:
            chunk_index: Index of the chunk to load
            
        Returns:
            List of road segments or None if chunk doesn't exist
        """
        chunk_file = self.chunk_dir / f"chunk_{chunk_index:04d}.pkl"
        
        if not chunk_file.exists():
            return None
        
        try:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            return chunk_data.get('chunk_data', [])
        except Exception as e:
            print(f"Error loading chunk {chunk_index}: {e}")
            return None
    
    def save_progress(self, processed_chunks: List[int], total_chunks: int, metadata: Dict):
        """
        Save analysis progress to disk.
        
        Args:
            processed_chunks: List of chunk indices that have been processed
            total_chunks: Total number of chunks in the analysis
            metadata: Analysis metadata
        """
        try:
            progress_data = {
                'processed_chunks': processed_chunks,
                'total_chunks': total_chunks,
                'timestamp': time.time()
            }
            
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def cleanup(self):
        """Remove all checkpoint files for this analysis."""
        try:
            import shutil
            if self.analysis_dir.exists():
                shutil.rmtree(self.analysis_dir)
                print(f"Cleaned up checkpoint files for analysis: {self.analysis_id}")
        except Exception as e:
            print(f"Error cleaning up files: {e}")
    
    def list_saved_chunks(self) -> List[int]:
        """Get list of chunk indices that have been saved."""
        if not self.chunk_dir.exists():
            return []
        
        chunk_indices = []
        for chunk_file in self.chunk_dir.glob("chunk_*.pkl"):
            try:
                # Extract chunk index from filename
                index_str = chunk_file.stem.split('_')[1]
                chunk_indices.append(int(index_str))
            except (ValueError, IndexError):
                continue
        
        return sorted(chunk_indices)
    
    # NEW ELEVATION CHECKPOINT METHODS
    
    def save_elevation_progress(self, coordinate_mapping: Dict, batch_info: Dict):
        """
        Save elevation fetching progress silently (no print statements to avoid breaking progress bar).
        
        Args:
            coordinate_mapping: Dict mapping coordinates to elevations
            batch_info: Info about current batch progress
        """
        try:
            elevation_data = {
                'coordinate_mapping': coordinate_mapping,
                'batch_info': batch_info,
                'timestamp': time.time()
            }
            
            with open(self.elevation_progress_file, 'wb') as f:
                pickle.dump(elevation_data, f)
                
        except Exception as e:
            # Don't print errors during elevation fetching to avoid breaking progress bar
            pass
    
    def load_elevation_progress(self) -> Tuple[Dict, Dict]:
        """
        Load elevation progress from disk.
        
        Returns:
            (coordinate_mapping, batch_info)
        """
        if not self.elevation_progress_file.exists():
            return {}, {}
        
        try:
            with open(self.elevation_progress_file, 'rb') as f:
                elevation_data = pickle.load(f)
            
            return elevation_data.get('coordinate_mapping', {}), elevation_data.get('batch_info', {})
        except Exception as e:
            print(f"Error loading elevation progress: {e}")
            return {}, {}
    
    def clear_elevation_progress(self):
        """Clear elevation progress after successful completion."""
        try:
            if self.elevation_progress_file.exists():
                self.elevation_progress_file.unlink()
        except Exception as e:
            print(f"Error clearing elevation progress: {e}")

class ElevationFetcher:
    """Elevation data fetcher with checkpoint support."""

    def __init__(self, url=TOPO_API_URL, batch_size: int = 500, delay_between_requests: float = 1.0):
        self.base_url = TOPO_API_URL
        self.batch_size = batch_size
        self.delay_between_requests = delay_between_requests

    def fetch_elevations_for_coordinates_with_checkpoints(self, coordinates: List[Tuple[float, float]], 
                                                        persistence: ChunkPersistenceManager,
                                                        progress_desc: str = "Fetching elevation") -> List[Optional[float]]:
        """
        Memory-efficient elevation fetching with checkpoint support.
        """
        if not coordinates:
            return []

        # Check for existing elevation progress
        existing_mapping, batch_info = persistence.load_elevation_progress()
        
        if existing_mapping:
            print(f"Found existing elevation progress: {len(existing_mapping)} coordinates completed")
            resume_choice = input("Resume elevation fetching from checkpoint? (y/n): ").strip().lower()
            if resume_choice == 'y':
                return self._resume_elevation_fetching(coordinates, existing_mapping, batch_info, 
                                                     persistence, progress_desc)
        
        # Start fresh elevation fetching
        return self._fetch_elevations_with_checkpoints(coordinates, persistence, progress_desc)
    
    def _fetch_elevations_with_checkpoints(self, coordinates: List[Tuple[float, float]], 
                                         persistence: ChunkPersistenceManager,
                                         progress_desc: str) -> List[Optional[float]]:
        """Fetch elevations with incremental checkpoint saving."""
        
        # Deduplicate coordinates while preserving order
        unique_coords = []
        coord_to_index = {}
        original_to_unique = []
        
        for i, coord in enumerate(coordinates):
            rounded_coord = (round(coord[0], 6), round(coord[1], 6))
            
            if rounded_coord not in coord_to_index:
                coord_to_index[rounded_coord] = len(unique_coords)
                unique_coords.append(rounded_coord)
            
            original_to_unique.append(coord_to_index[rounded_coord])

        total_unique = len(unique_coords)
        total_original = len(coordinates)
        
        print(f"Fetching elevation data for {total_unique} unique coordinates (deduplicated from {total_original})")

        # Process unique coordinates in batches with checkpointing
        all_elevations = [None] * total_unique
        coordinate_mapping = {}  # Track coord -> elevation mapping for checkpoints
        total_batches = (total_unique + self.batch_size - 1) // self.batch_size
        
        checkpoint_interval = max(1, total_batches // 10)  # Save every 10% of batches
        
        with tqdm(total=total_batches, desc=progress_desc, unit="batch", mininterval=1.0) as pbar:
            
            for batch_num in range(0, total_unique, self.batch_size):
                batch_coords = unique_coords[batch_num:batch_num + self.batch_size]
                batch_index = batch_num // self.batch_size

                try:
                    batch_elevations = self._fetch_batch_elevations(batch_coords)
                    
                    # Store results and update mapping
                    for j, elevation in enumerate(batch_elevations):
                        coord_idx = batch_num + j
                        if coord_idx < total_unique:
                            all_elevations[coord_idx] = elevation
                            if elevation is not None:
                                coordinate_mapping[unique_coords[coord_idx]] = elevation

                    successful_elevations = sum(1 for e in batch_elevations if e is not None)
                    
                    pbar.set_postfix({
                        'unique_coords': f"{min(batch_num + self.batch_size, total_unique)}/{total_unique}",
                        'success_rate': f"{successful_elevations}/{len(batch_coords)}",
                        'checkpoints': len(coordinate_mapping)
                    })
                    
                    # Save checkpoint periodically
                    if batch_index % checkpoint_interval == 0 or batch_index == total_batches - 1:
                        batch_info = {
                            'completed_batches': batch_index + 1,
                            'total_batches': total_batches,
                            'last_batch_index': batch_num,
                            'total_unique': total_unique
                        }
                        persistence.save_elevation_progress(coordinate_mapping, batch_info)
                    
                    # Be respectful to the API
                    if batch_num + self.batch_size < total_unique:
                        time.sleep(self.delay_between_requests)

                except Exception as e:
                    print(f"  Error in batch {batch_index + 1}: {e}")
                    # Save checkpoint before failing
                    batch_info = {
                        'completed_batches': batch_index,
                        'total_batches': total_batches,
                        'last_batch_index': batch_num,
                        'total_unique': total_unique,
                        'error': str(e)
                    }
                    persistence.save_elevation_progress(coordinate_mapping, batch_info)
                    
                    # Fill with None for failed batch
                    for j in range(len(batch_coords)):
                        coord_idx = batch_num + j
                        if coord_idx < total_unique:
                            all_elevations[coord_idx] = None
                    
                pbar.update(1)

        # Map back to original coordinate order
        result_elevations = []
        for unique_idx in original_to_unique:
            result_elevations.append(all_elevations[unique_idx])

        successful_total = sum(1 for e in result_elevations if e is not None)
        print(f"Successfully fetched elevation data for {successful_total}/{total_original} coordinates")
        
        # Clear checkpoint after successful completion
        persistence.clear_elevation_progress()
        
        # Clean up memory
        del unique_coords, coord_to_index, original_to_unique, all_elevations, coordinate_mapping
        gc.collect()
        
        return result_elevations
    
    def _resume_elevation_fetching(self, coordinates: List[Tuple[float, float]], 
                                 existing_mapping: Dict, batch_info: Dict,
                                 persistence: ChunkPersistenceManager,
                                 progress_desc: str) -> List[Optional[float]]:
        """Resume elevation fetching from checkpoint."""
        
        print(f"Resuming elevation fetching from batch {batch_info.get('completed_batches', 0)}/{batch_info.get('total_batches', 0)}")
        
        # Rebuild coordinate structure
        unique_coords = []
        coord_to_index = {}
        original_to_unique = []
        
        for i, coord in enumerate(coordinates):
            rounded_coord = (round(coord[0], 6), round(coord[1], 6))
            
            if rounded_coord not in coord_to_index:
                coord_to_index[rounded_coord] = len(unique_coords)
                unique_coords.append(rounded_coord)
            
            original_to_unique.append(coord_to_index[rounded_coord])

        total_unique = len(unique_coords)
        all_elevations = [None] * total_unique
        
        # Fill in existing elevations
        for i, coord in enumerate(unique_coords):
            if coord in existing_mapping:
                all_elevations[i] = existing_mapping[coord]
        
        existing_count = sum(1 for e in all_elevations if e is not None)
        remaining_count = total_unique - existing_count
        
        print(f"Loaded {existing_count} existing elevations, {remaining_count} remaining to fetch")
        
        if remaining_count == 0:
            print("All elevations already completed!")
            # Map back to original order
            result_elevations = []
            for unique_idx in original_to_unique:
                result_elevations.append(all_elevations[unique_idx])
            return result_elevations
        
        # Continue fetching missing elevations
        last_batch_index = batch_info.get('last_batch_index', 0)
        total_batches = (total_unique + self.batch_size - 1) // self.batch_size
        remaining_batches = total_batches - (last_batch_index // self.batch_size)
        
        checkpoint_interval = max(1, remaining_batches // 10)
        coordinate_mapping = existing_mapping.copy()
        
        with tqdm(total=remaining_batches, desc=f"{progress_desc} (resumed)", unit="batch", mininterval=1.0) as pbar:
            
            batch_counter = 0
            for batch_num in range(last_batch_index, total_unique, self.batch_size):
                batch_coords = unique_coords[batch_num:batch_num + self.batch_size]
                
                # Skip coordinates we already have
                coords_to_fetch = []
                coord_indices = []
                for j, coord in enumerate(batch_coords):
                    coord_idx = batch_num + j
                    if coord_idx < total_unique and all_elevations[coord_idx] is None:
                        coords_to_fetch.append(coord)
                        coord_indices.append(coord_idx)
                
                if not coords_to_fetch:
                    pbar.update(1)
                    batch_counter += 1
                    continue

                try:
                    batch_elevations = self._fetch_batch_elevations(coords_to_fetch)
                    
                    # Store results
                    for idx, elevation in zip(coord_indices, batch_elevations):
                        all_elevations[idx] = elevation
                        if elevation is not None:
                            coordinate_mapping[unique_coords[idx]] = elevation

                    successful_elevations = sum(1 for e in batch_elevations if e is not None)
                    
                    pbar.set_postfix({
                        'fetched': f"{len(coords_to_fetch)}",
                        'success_rate': f"{successful_elevations}/{len(coords_to_fetch)}",
                        'total_completed': len(coordinate_mapping)
                    })
                    
                    # Save checkpoint periodically
                    if batch_counter % checkpoint_interval == 0 or batch_num + self.batch_size >= total_unique:
                        batch_info_updated = {
                            'completed_batches': (batch_num // self.batch_size) + 1,
                            'total_batches': total_batches,
                            'last_batch_index': batch_num + self.batch_size,
                            'total_unique': total_unique
                        }
                        persistence.save_elevation_progress(coordinate_mapping, batch_info_updated)
                    
                    # Be respectful to the API
                    if coords_to_fetch and batch_num + self.batch_size < total_unique:
                        time.sleep(self.delay_between_requests)

                except Exception as e:
                    print(f"  Error in resumed batch: {e}")
                    # Save checkpoint before failing
                    batch_info_updated = {
                        'completed_batches': batch_num // self.batch_size,
                        'total_batches': total_batches,
                        'last_batch_index': batch_num,
                        'total_unique': total_unique,
                        'error': str(e)
                    }
                    persistence.save_elevation_progress(coordinate_mapping, batch_info_updated)
                    raise
                    
                pbar.update(1)
                batch_counter += 1

        # Map back to original coordinate order
        result_elevations = []
        for unique_idx in original_to_unique:
            result_elevations.append(all_elevations[unique_idx])

        successful_total = sum(1 for e in result_elevations if e is not None)
        print(f"Successfully fetched elevation data for {successful_total}/{len(coordinates)} coordinates")
        
        # Clear checkpoint after successful completion
        persistence.clear_elevation_progress()
        
        # Clean up memory
        del unique_coords, coord_to_index, original_to_unique, all_elevations, coordinate_mapping
        gc.collect()
        
        return result_elevations

    def _fetch_batch_elevations(self, coordinates: List[Tuple[float, float]]) -> List[Optional[float]]:
        """Fetch elevation data for a batch of coordinates."""
        locations_str = "|".join([f"{lat},{lon}" for lat, lon in coordinates])
        params = {'locations': locations_str}

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'OK':
                print(f"    API returned status: {data.get('status')}")
                return [None] * len(coordinates)

            elevations = []
            results = data.get('results', [])

            for i, result in enumerate(results):
                if result and result.get('elevation') is not None:
                    elevations.append(float(result['elevation']))
                else:
                    elevations.append(None)

            while len(elevations) < len(coordinates):
                elevations.append(None)

            return elevations[:len(coordinates)]

        except requests.exceptions.RequestException as e:
            print(f"    Request failed: {e}")
            return [None] * len(coordinates)
        except (ValueError, KeyError) as e:
            print(f"    Failed to parse response: {e}")
            return [None] * len(coordinates)


class ChunkedRoadNetworkAnalyzer:
    """Memory-efficient road network analyzer that processes data in chunks."""

    def __init__(self, surface_filter: str = 'all', chunk_size_km: float = 5.0):
        self.geolocator = Nominatim(user_agent="climb_analyzer")
        if OVERPASS_API_URL:
            self.overpass_api = overpy.Overpass(url=OVERPASS_API_URL)
        else:
            self.overpass_api = overpy.Overpass()            
        self.surface_filter = surface_filter
        self.chunk_size_km = chunk_size_km

    def get_coordinates_from_address(self, address: str, country: str = None) -> Tuple[float, float, str]:
        """Convert street address to coordinates with fallback strategies"""
        try:
            if country:
                search_query = f"{address}, {country}"
            else:
                search_query = address
            
            print(f"Geocoding address: {search_query}")
            location = self.geolocator.geocode(search_query, timeout=10)
            
            if location:
                formatted_address = location.address
                print(f"Found location: {formatted_address}")
                print(f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}")
                return location.latitude, location.longitude, formatted_address
            
            # Fallback strategies if full address fails
            print(f"Full address not found. Trying fallback strategies...")
            
            # Strategy 1: Try without house number
            address_parts = address.split(',')
            if len(address_parts) > 1:
                # Remove the first part (likely house number + street)
                street_part = address_parts[0].strip()
                # Try to extract just the street name (remove house number)
                street_words = street_part.split()
                if len(street_words) > 1 and street_words[0].isdigit():
                    street_name = ' '.join(street_words[1:])
                    fallback_address = ', '.join([street_name] + address_parts[1:])
                    if country:
                        fallback_query = f"{fallback_address}, {country}"
                    else:
                        fallback_query = fallback_address
                    
                    print(f"Trying street name only: {fallback_query}")
                    location = self.geolocator.geocode(fallback_query, timeout=10)
                    if location:
                        print(f"Found using street name: {location.address}")
                        print(f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}")
                        return location.latitude, location.longitude, location.address
            
            # Strategy 2: Try just city, state, zip
            if len(address_parts) >= 2:
                city_state_zip = ', '.join(address_parts[1:]).strip()
                if country:
                    fallback_query = f"{city_state_zip}, {country}"
                else:
                    fallback_query = city_state_zip
                
                print(f"Trying city/state/zip: {fallback_query}")
                location = self.geolocator.geocode(fallback_query, timeout=10)
                if location:
                    print(f"Found using city/state: {location.address}")
                    print(f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}")
                    print("Note: Using city center coordinates since exact address wasn't found")
                    return location.latitude, location.longitude, location.address
            
            # Strategy 3: Manual coordinate entry
            print(f"\nGeocoding failed for address: {search_query}")
            print("You can:")
            print("1. Enter coordinates manually")
            print("2. Try a simpler address (just city, state)")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                try:
                    lat = float(input("Enter latitude (decimal degrees): ").strip())
                    lon = float(input("Enter longitude (decimal degrees): ").strip())
                    manual_address = input("Enter description for this location: ").strip() or "Manual coordinates"
                    print(f"Using manual coordinates: {lat:.6f}, {lon:.6f}")
                    return lat, lon, manual_address
                except ValueError:
                    raise ValueError("Invalid coordinates entered")
            elif choice == '2':
                new_address = input("Enter a simpler address: ").strip()
                if new_address:
                    return self.get_coordinates_from_address(new_address, country)
                else:
                    raise ValueError("No address provided")
            else:
                raise ValueError("Geocoding cancelled by user")
                
        except Exception as e:
            if "cancelled by user" in str(e) or "Invalid coordinates" in str(e):
                raise ValueError(str(e))
            else:
                raise ValueError(f"Error geocoding address: {e}")

    def calculate_chunks(self, center_lat: float, center_lon: float, radius_km: float) -> List[Tuple[float, float, float]]:
        """
        Calculate geographic chunks to process data in smaller pieces.
        Returns list of (lat, lon, chunk_radius) tuples.
        """
        if radius_km <= self.chunk_size_km:
            return [(center_lat, center_lon, radius_km)]
        
        chunks = []
        # Simple grid-based chunking
        chunk_radius = self.chunk_size_km / 2
        
        # Calculate how many chunks we need in each direction
        chunks_per_side = math.ceil(radius_km / self.chunk_size_km)
        
        for i in range(-chunks_per_side, chunks_per_side + 1):
            for j in range(-chunks_per_side, chunks_per_side + 1):
                # Calculate chunk center
                lat_offset = i * self.chunk_size_km / 111.0  # Rough km to degree conversion
                lon_offset = j * self.chunk_size_km / (111.0 * math.cos(math.radians(center_lat)))
                
                chunk_lat = center_lat + lat_offset
                chunk_lon = center_lon + lon_offset
                
                # Check if chunk is within the original radius
                distance = self.calculate_distance(center_lat, center_lon, chunk_lat, chunk_lon)
                if distance <= radius_km:
                    chunks.append((chunk_lat, chunk_lon, chunk_radius))
        
        print(f"Processing area in {len(chunks)} chunks of ~{self.chunk_size_km}km radius each")
        return chunks

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def build_overpass_query(self, center_lat: float, center_lon: float, radius_km: float) -> str:
        """Build memory-efficient Overpass query with surface filtering."""
        
        # Add distance filter to reduce unnecessary data
        distance_filter = f"(around:{radius_km * 1000},{center_lat},{center_lon})"
        
        if self.surface_filter == 'paved':
            query = f"""
            [out:json][timeout:60][maxsize:1073741824];
            (
              way[highway~"trunk|primary|secondary|tertiary|unclassified|residential|service"]
                  [surface!~"unpaved|gravel|dirt|sand|grass|ground|earth|mud|clay"]
                  {distance_filter};
            );
            (._;>;);
            out geom;
            """
        elif self.surface_filter == 'gravel':
            query = f"""
            [out:json][timeout:60][maxsize:1073741824];
            (
              way[highway~"track|unclassified|tertiary|residential|service"]
                  [surface~"gravel|compacted|fine_gravel"]
                  {distance_filter};
            );
            (._;>;);
            out geom;
            """
        elif self.surface_filter == 'dirt':
            query = f"""
            [out:json][timeout:60][maxsize:1073741824];
            (
              way[highway~"track|path"]
                  {distance_filter};
            );
            (._;>;);
            out geom;
            """
        else:  # 'all'
            query = f"""
            [out:json][timeout:60][maxsize:1073741824];
            (
              way[highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|track|path"]
                  {distance_filter};
            );
            (._;>;);
            out geom;
            """
        
        return query

    def get_roads_in_chunk(self, chunk_lat: float, chunk_lon: float, chunk_radius: float) -> List:
        """Get road network data for a single chunk."""
        try:
            query = self.build_overpass_query(chunk_lat, chunk_lon, chunk_radius)
            
            local_api_url = "http://localhost:12345/api/interpreter"
            
            response = requests.post(
                local_api_url,
                data=query,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"HTTP Error {response.status_code} for chunk ({chunk_lat:.4f}, {chunk_lon:.4f})")
                return []
            
            response_text = response.text
            json_start = response_text.find('{')
            if json_start == -1:
                return []
            
            json_text = response_text[json_start:]
            
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed for chunk: {e}")
                return []
            
            # Process ways with memory efficiency
            ways = []
            
            # Build node lookup only for nodes we need
            all_nodes = {}
            for element in data.get('elements', []):
                if element.get('type') == 'node':
                    all_nodes[element['id']] = element
            
            # Create ways
            for element in data.get('elements', []):
                if element.get('type') == 'way':
                    way = self._create_simple_way(element, all_nodes)
                    if way and len(way.nodes) > 1:
                        ways.append(way)
            
            # Clean up memory
            del data, all_nodes
            gc.collect()
            
            return ways
            
        except Exception as e:
            print(f"Error processing chunk ({chunk_lat:.4f}, {chunk_lon:.4f}): {e}")
            return []

    def _create_simple_way(self, element, all_nodes):
        """Create a simple way object efficiently."""
        way_id = element['id']
        tags = element.get('tags', {})
        nodes = []
        
        # Handle geometry data (preferred for memory efficiency)
        if 'geometry' in element:
            for idx, geom_point in enumerate(element['geometry']):
                if 'lat' in geom_point and 'lon' in geom_point:
                    node_data = {
                        'id': f"{way_id}_{idx}",
                        'lat': geom_point['lat'],
                        'lon': geom_point['lon']
                    }
                    node = SimpleNode(node_data)
                    nodes.append(node)
        
        # Handle regular node references (fallback)
        elif 'nodes' in element:
            for node_id in element['nodes']:
                if node_id in all_nodes:
                    node = SimpleNode(all_nodes[node_id])
                    nodes.append(node)
        
        if nodes:
            way = SimpleWay()
            way.id = way_id
            way.tags = tags
            way.nodes = nodes
            return way
        
        return None

    def get_roads_in_area_chunked(self, center_lat: float, center_lon: float, radius_unit='miles', radius_km: float = 15) -> Generator:
        """
        Get road network data in chunks to manage memory usage.
        Returns a generator of way lists.
        """
        print(f"Analyzing area with chunked processing...")
        chunks = self.calculate_chunks(center_lat, center_lon, radius_km)
        
        total_ways = 0
        
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            for i, (chunk_lat, chunk_lon, chunk_radius) in enumerate(chunks):
                chunk_ways = self.get_roads_in_chunk(chunk_lat, chunk_lon, chunk_radius)
                
                if chunk_ways:
                    total_ways += len(chunk_ways)
                    yield chunk_ways
                
                pbar.set_postfix({
                    'total_ways': total_ways,
                    'chunk': f"{i+1}/{len(chunks)}"
                })
                pbar.update(1)
                
                # Small delay to be respectful to the API
                time.sleep(0.5)

class SimpleWay:
    """Lightweight way object."""
    def __init__(self):
        self.id = None
        self.tags = {}
        self.nodes = []

class SimpleNode:
    """Lightweight node object."""
    def __init__(self, node_data):
        self.id = node_data.get('id', 0)
        self.lat = float(node_data.get('lat', 0.0))
        self.lon = float(node_data.get('lon', 0.0))

class CrossChunkRoadMerger:
    """Handles merging of road segments that cross chunk boundaries."""
    
    def __init__(self, coordinate_tolerance: float = 0.002):  # Increased tolerance
        """
        Initialize cross-chunk merger with more permissive tolerance.
        0.002 degrees ≈ 222m - more realistic for road connections
        """
        self.coordinate_tolerance = coordinate_tolerance
        self.distance_tolerance_m = 200  # 200m max gap
        
    def _coordinates_match(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> bool:
        """Coordinate matching using actual distance."""
        # First try the original coordinate difference method
        if (abs(coord1[0] - coord2[0]) < self.coordinate_tolerance and
            abs(coord1[1] - coord2[1]) < self.coordinate_tolerance):
            return True
        
        # If that fails, try distance-based matching  
        distance_km = self.calculate_distance(coord1[0], coord1[1], coord2[0], coord2[1])
        distance_m = distance_km * 1000
        
        return distance_m < self.distance_tolerance_m
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        import math
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c
        
    def merge_cross_chunk_segments(self, all_segments: List[Dict]) -> List[Dict]:
        """
        Merge segments that represent the same road across different chunks.
        Handles long roads like Mt Mitchell Rd that span multiple chunks.
        """
        print(f"Performing cross-chunk merge on {len(all_segments)} segments...")
        
        # Group segments by street name for efficient processing
        segments_by_name = defaultdict(list)
        for segment in all_segments:
            street_name = segment.get('street_name', '').strip()
            if street_name:
                segments_by_name[street_name].append(segment)
        
        merged_segments = []
        processed_segment_ids = set()
        
        # Process each street name group
        for street_name, street_segments in segments_by_name.items():
            if len(street_segments) == 1:
                # Only one segment for this street, no merging needed
                merged_segments.extend(street_segments)
                continue
            
            # Find connected segments for this street
            merged_street_segments = self._merge_segments_for_street(street_segments, street_name)
            merged_segments.extend(merged_street_segments)
        
        print(f"Cross-chunk merge completed: {len(all_segments)} -> {len(merged_segments)} segments")
        return merged_segments
    
    def _merge_segments_for_street(self, segments: List[Dict], street_name: str) -> List[Dict]:
        """Merge segments for a specific street name."""
        if len(segments) <= 1:
            return segments
        
        merged_segments = []
        used_segments = set()
        
        # Sort segments by length (longest first) to prioritize merging longer segments
        segments.sort(key=lambda s: len(s.get('nodes', [])), reverse=True)
        
        for i, segment in enumerate(segments):
            if i in used_segments:
                continue
            
            # Start with this segment
            current_merged = {
                'way_ids': segment['way_ids'].copy(),
                'nodes': segment['nodes'].copy(),
                'street_name': segment['street_name'],
                'surface': segment['surface'],
                'tracktype': segment['tracktype'],
                'tracktype_definition': segment['tracktype_definition']
            }
            used_segments.add(i)
            
            # Try to extend this segment by connecting other segments
            extended = True
            iteration = 0
            max_iterations = len(segments) * 3  # Increased persistence

            while extended and iteration < max_iterations:
                extended = False
                iteration += 1
                
                for j, other_segment in enumerate(segments):
                    if j in used_segments:
                        continue
                    
                    # Check if segments can be connected
                    connection_type = self._check_segment_connectivity(current_merged, other_segment)
                    
                    if connection_type:
                        # Merge the segments
                        current_merged = self._connect_segments(current_merged, other_segment, connection_type)
                        used_segments.add(j)
                        extended = True
                        break

            
            merged_segments.append(current_merged)
        
        return merged_segments
    
    def _check_segment_connectivity(self, segment1: Dict, segment2: Dict) -> Optional[str]:
        """
        Check if two segments can be connected.
        Returns connection type or None if not connectable.
        """
        nodes1 = segment1.get('nodes', [])
        nodes2 = segment2.get('nodes', [])
        
        if not nodes1 or not nodes2:
            return None
        
        # Get endpoint coordinates
        start1 = self._get_node_coordinates(nodes1[0])
        end1 = self._get_node_coordinates(nodes1[-1])
        start2 = self._get_node_coordinates(nodes2[0])
        end2 = self._get_node_coordinates(nodes2[-1])
        
        # Check all possible connections
        if self._coordinates_match(end1, start2):
            return 'end1_to_start2'  # segment1 end -> segment2 start
        elif self._coordinates_match(end1, end2):
            return 'end1_to_end2'    # segment1 end -> segment2 end (reverse segment2)
        elif self._coordinates_match(start1, start2):
            return 'start1_to_start2' # segment1 start -> segment2 start (reverse segment1)
        elif self._coordinates_match(start1, end2):
            return 'start1_to_end2'   # segment1 start -> segment2 end
        
        return None
    
    def _get_node_coordinates(self, node) -> Tuple[float, float]:
        """Extract coordinates from a node."""
        if hasattr(node, 'lat') and hasattr(node, 'lon'):
            return (float(node.lat), float(node.lon))
        return (0.0, 0.0)
    
    def _coordinates_match(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> bool:
        """Check if two coordinates match within tolerance."""
        return (abs(coord1[0] - coord2[0]) < self.coordinate_tolerance and
                abs(coord1[1] - coord2[1]) < self.coordinate_tolerance)
    
    def _connect_segments(self, segment1: Dict, segment2: Dict, connection_type: str) -> Dict:
        """Connect two segments based on connection type."""
        nodes1 = segment1['nodes'].copy()
        nodes2 = segment2['nodes'].copy()
        
        if connection_type == 'end1_to_start2':
            # segment1 + segment2 (skip duplicate node)
            connected_nodes = nodes1 + nodes2[1:]
        elif connection_type == 'end1_to_end2':
            # segment1 + reversed segment2 (skip duplicate node)
            connected_nodes = nodes1 + nodes2[-2::-1]
        elif connection_type == 'start1_to_start2':
            # reversed segment1 + segment2 (skip duplicate node)
            connected_nodes = nodes1[::-1] + nodes2[1:]
        elif connection_type == 'start1_to_end2':
            # segment2 + segment1 (skip duplicate node)
            connected_nodes = nodes2 + nodes1[1:]
        else:
            # Fallback: just combine nodes
            connected_nodes = nodes1 + nodes2
        
        # Create merged segment
        merged_segment = {
            'way_ids': segment1['way_ids'] + segment2['way_ids'],
            'nodes': connected_nodes,
            'street_name': segment1['street_name'],
            'surface': segment1['surface'],
            'tracktype': segment1['tracktype'],
            'tracktype_definition': segment1['tracktype_definition']
        }
        
        # Update surface/tracktype if current is unknown but other has info
        if segment1['surface'] == 'unknown' and segment2['surface'] != 'unknown':
            merged_segment['surface'] = segment2['surface']
        
        if (segment1['tracktype'] in ['unknown', 'not_applicable'] and 
            segment2['tracktype'] not in ['unknown', 'not_applicable']):
            merged_segment['tracktype'] = segment2['tracktype']
            merged_segment['tracktype_definition'] = segment2['tracktype_definition']
        
        return merged_segment

class MemoryEfficientMerger:
    """Memory-efficient road segment merger."""
    
    def __init__(self):
        self.way_lookup = {}  # For deduplication
        
    def process_way_chunk(self, ways_chunk: List) -> List[Dict]:
        """Process a chunk of ways and return merged segments."""
        # Deduplicate ways by ID to avoid processing the same way multiple times
        unique_ways = {}
        for way in ways_chunk:
            if hasattr(way, 'id') and way.id not in unique_ways:
                unique_ways[way.id] = way
        
        # Group by street name
        street_groups = defaultdict(list)
        for way in unique_ways.values():
            street_name = self._get_street_name(way)
            if street_name:
                street_groups[street_name].append(way)
        
        # Merge within groups
        merged_segments = []
        for street_name, ways_group in street_groups.items():
            segments = self._merge_ways_in_group(ways_group)
            merged_segments.extend(segments)
        
        # Clean up
        del unique_ways, street_groups
        gc.collect()

        for segment in merged_segments:
            # Add start coordinates for efficient lookup later
            if segment['nodes'] and len(segment['nodes']) > 0:
                first_node = segment['nodes'][0]
                if hasattr(first_node, 'lat') and hasattr(first_node, 'lon'):
                    segment['start_lat'] = float(first_node.lat)
                    segment['start_lon'] = float(first_node.lon)
    
        return merged_segments

    def _get_street_name(self, way) -> Optional[str]:
        """Extract street name efficiently."""
        if not hasattr(way, 'tags') or not way.tags:
            return None
        
        name_tags = ['name', 'ref', 'addr:street', 'tiger:name_base', 'unsigned_ref', 'highway']
        
        for tag in name_tags:
            if tag in way.tags and way.tags[tag]:
                return str(way.tags[tag]).strip()
        
        return None

    def _merge_ways_in_group(self, ways_group: List) -> List[Dict]:
        """Efficiently merge ways in a group."""
        if not ways_group:
            return []
        
        merged_segments = []
        used_ways = set()
        
        for start_way in ways_group:
            if start_way.id in used_ways:
                continue
            
            # Create merged segment
            segment = {
                'way_ids': [start_way.id],
                'nodes': start_way.nodes.copy() if start_way.nodes else [],
                'street_name': self._get_street_name(start_way),
                'surface': self._get_surface(start_way),
                'tracktype': self._get_tracktype(start_way)[0],
                'tracktype_definition': self._get_tracktype(start_way)[1]
            }
            used_ways.add(start_way.id)
            
            # Try to extend (simplified merging for memory efficiency)
            extended = True
            while extended and len(used_ways) < len(ways_group):
                extended = False
                
                for way2 in ways_group:
                    if way2.id in used_ways:
                        continue
                    
                    # Simple endpoint connection check
                    if self._can_connect_simple(segment['nodes'], way2.nodes):
                        segment['nodes'].extend(way2.nodes[1:])  # Simple append, skip duplicate
                        segment['way_ids'].append(way2.id)
                        used_ways.add(way2.id)
                        extended = True
                        break
            
            merged_segments.append(segment)
        
        return merged_segments

    def _can_connect_simple(self, segment_nodes: List, way_nodes: List) -> bool:
        """Simple connection check for memory efficiency."""
        if not segment_nodes or not way_nodes:
            return False
        
        # Check if last node of segment connects to first node of way
        if hasattr(segment_nodes[-1], 'id') and hasattr(way_nodes[0], 'id'):
            return segment_nodes[-1].id == way_nodes[0].id
        
        return False

    def _get_surface(self, way) -> str:
        """Extract surface information."""
        if not hasattr(way, 'tags') or not way.tags:
            return "unknown"
        
        if 'surface' in way.tags:
            return str(way.tags['surface']).strip().lower()
        
        highway_type = way.tags.get('highway', '').strip().lower()
        surface_mapping = {
            'motorway': 'asphalt', 'trunk': 'asphalt', 'primary': 'asphalt',
            'secondary': 'asphalt', 'tertiary': 'asphalt', 'unclassified': 'paved',
            'residential': 'asphalt', 'service': 'paved', 'track': 'unpaved',
            'path': 'unpaved', 'footway': 'unpaved'
        }
        return surface_mapping.get(highway_type, "unknown")

    def _get_tracktype(self, way) -> Tuple[str, str]:
        """Extract tracktype information."""
        if not hasattr(way, 'tags') or not way.tags:
            return "unknown", "No tracktype information"
        
        if way.tags.get('highway') != 'track':
            return "not_applicable", "Not a track"
        
        if 'tracktype' in way.tags:
            tracktype = str(way.tags['tracktype']).strip().lower()
            definition = TRACKTYPE_DEFINITIONS.get(tracktype, f"Unknown tracktype: {tracktype}")
            return tracktype, definition
        
        return "unspecified", "Track with unspecified surface quality"

def find_existing_analysis(base_id: str, base_dir: Path) -> Optional[str]:
    """Find existing analysis with similar parameters."""
    if not base_dir.exists():
        return None
        
    matching_analyses = []
    for analysis_dir in base_dir.iterdir():
        if analysis_dir.is_dir() and analysis_dir.name.startswith(base_id + '_'):
            progress_file = analysis_dir / "progress.pkl"
            elevation_progress_file = analysis_dir / "elevation_progress.pkl"  # NEW: Check elevation too
            
            if progress_file.exists():
                try:
                    # Load chunk progress
                    with open(progress_file, 'rb') as f:
                        progress_data = pickle.load(f)
                    
                    completed = len(progress_data.get('processed_chunks', []))
                    total = progress_data.get('total_chunks', 0)
                    
                    # Load elevation progress
                    elevation_coords = 0
                    if elevation_progress_file.exists():
                        try:
                            with open(elevation_progress_file, 'rb') as f:
                                elevation_data = pickle.load(f)
                            elevation_coords = len(elevation_data.get('coordinate_mapping', {}))
                        except:
                            elevation_coords = 0
                                        
                    # Consider analysis resumable if:
                    # 1. Chunks are incomplete OR
                    # 2. Chunks are complete but elevation is in progress
                    chunks_incomplete = total > 0 and completed < total
                    elevation_in_progress = elevation_coords > 0
                    
                    if chunks_incomplete or elevation_in_progress:
                        # Calculate percentage based on phase
                        if chunks_incomplete:
                            percentage = (completed / total) * 100 if total > 0 else 0.0
                            phase = "chunk_processing"
                        else:
                            percentage = 100.0  # Chunks complete, elevation in progress
                            phase = "elevation_fetching"
                        
                        matching_analyses.append({
                            'id': analysis_dir.name,
                            'completed': completed,
                            'total': total,
                            'percentage': percentage,
                            'elevation_coords': elevation_coords,
                            'phase': phase
                        })
                        
                except Exception as e:
                    print(f"ERROR: reading progress for {analysis_dir.name}: {e}")
                    continue
    
    if matching_analyses:
        # Return the most recent incomplete analysis
        matching_analyses.sort(key=lambda x: x['id'], reverse=True)
        best_match = matching_analyses[0]
        
        print(f"\nFound existing resumable analysis:")
        print(f"   Analysis ID: {best_match['id']}")
        
        if best_match['phase'] == 'chunk_processing':
            print(f"   Chunk Progress: {best_match['completed']}/{best_match['total']} chunks ({best_match['percentage']:.1f}%)")
        else:
            print(f"   Chunks: {best_match['completed']}/{best_match['total']} (Complete)")
            print(f"   Elevation Progress: {best_match['elevation_coords']} coordinates")
        
        resume_choice = input(f"\nResume this analysis? (y/n): ").strip().lower()
        if resume_choice == 'y':
            return best_match['id']
    else:
        print("INFO: No resumable analyses found")
    
    return None

def resume_chunk_processing(analysis_id: str, min_score: float, unit_system: str):
    """Resume chunk processing from where it left off."""
    
    
    persistence = ChunkPersistenceManager(analysis_id)
    processed_chunks, total_chunks, metadata = persistence.load_progress()
    elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()
    
    print(f"  - Processed chunks: {len(processed_chunks)}/{total_chunks}")
    print(f"  - Elevation mapping: {len(elevation_mapping)} coordinates")
    
    # Check if all chunks are complete
    if len(processed_chunks) >= total_chunks:
        print("All chunks complete, checking elevation status...")
        
        if elevation_mapping:
            print(f"Found elevation progress.")
            print(f"  - Elevation coordinates: {len(elevation_mapping)}")
            print(f"  - Batch info: {elevation_batch_info}")
            
            # Load all chunk data
            all_merged_segments = []
            print("Loading all processed chunks...")
            with tqdm(total=len(processed_chunks), desc="Loading saved chunks", unit="chunk") as pbar:
                for chunk_index in processed_chunks:
                    chunk_data = persistence.load_chunk(chunk_index)
                    if chunk_data:
                        all_merged_segments.extend(chunk_data)
                    pbar.update(1)
            
            print(f"Loaded {len(all_merged_segments)} road segments from all chunks")
            
            # Initialize components
            elevation_fetcher = ElevationFetcher(batch_size=1000)
            cross_chunk_merger = CrossChunkRoadMerger(coordinate_tolerance=0.002)
            
            return complete_analysis_from_segments(all_merged_segments, elevation_fetcher,
                                                 cross_chunk_merger, metadata, 
                                                 min_score, unit_system, persistence)
        else:
            print("INFO: No elevation progress, starting fresh elevation analysis")
            
            # Load all chunk data and start elevation
            all_merged_segments = []
            print("Loading all processed chunks...")
            with tqdm(total=len(processed_chunks), desc="Loading saved chunks", unit="chunk") as pbar:
                for chunk_index in processed_chunks:
                    chunk_data = persistence.load_chunk(chunk_index)
                    if chunk_data:
                        all_merged_segments.extend(chunk_data)
                    pbar.update(1)
            
            print(f"Loaded {len(all_merged_segments)} road segments from all chunks")
            
            # Initialize components
            elevation_fetcher = ElevationFetcher(batch_size=1000)
            cross_chunk_merger = CrossChunkRoadMerger(coordinate_tolerance=0.002)
            
            return complete_analysis_from_segments(all_merged_segments, elevation_fetcher,
                                                 cross_chunk_merger, metadata, 
                                                 min_score, unit_system, persistence)
    
    # If chunks are not complete, continue chunk processing
    print(f"Continuing chunk processing: {len(processed_chunks)}/{total_chunks} completed")
    
    # Load existing chunks
    all_merged_segments = []
    print("Loading previously processed chunks...")
    with tqdm(total=len(processed_chunks), desc="Loading saved chunks", unit="chunk") as pbar:
        for chunk_index in processed_chunks:
            chunk_data = persistence.load_chunk(chunk_index)
            if chunk_data:
                all_merged_segments.extend(chunk_data)
            pbar.update(1)
    
    print(f"Loaded {len(all_merged_segments)} road segments from {len(processed_chunks)} completed chunks")
    
    # Continue processing remaining chunks
    remaining_chunks = total_chunks - len(processed_chunks)
    
    # Rebuild components and continue
    chunks = metadata.get('chunks', [])
    surface_filter = metadata.get('surface_filter', 'all')
    chunk_size_km = metadata.get('chunk_size_km', 8.0)
    
    road_analyzer = ChunkedRoadNetworkAnalyzer(surface_filter, chunk_size_km)
    chunk_merger = MemoryEfficientMerger()
    cross_chunk_merger = CrossChunkRoadMerger(coordinate_tolerance=0.002)
    elevation_fetcher = ElevationFetcher(batch_size=1000)
    
    return continue_chunk_processing(persistence, road_analyzer, chunk_merger, cross_chunk_merger,
                                   elevation_fetcher, chunks, metadata, all_merged_segments,
                                   processed_chunks, min_score, unit_system)




def continue_chunk_processing(persistence, road_analyzer, chunk_merger, cross_chunk_merger,
                            elevation_fetcher, chunks, metadata, existing_segments, 
                            processed_chunks, min_score, unit_system):
    """Continue processing chunks from where we left off."""
    
    total_chunks = len(chunks)
    processed_set = set(processed_chunks)
    all_merged_segments = existing_segments.copy()
    
    print("Continuing chunk processing with checkpoint saving...")
    
    remaining_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in processed_set]
    
    with tqdm(total=len(remaining_chunks), desc="Processing remaining chunks", unit="chunk") as pbar:
        for chunk_index, (chunk_lat, chunk_lon, chunk_radius) in remaining_chunks:
            
            try:
                chunk_ways = road_analyzer.get_roads_in_chunk(chunk_lat, chunk_lon, chunk_radius)
                
                if chunk_ways:
                    chunk_segments = chunk_merger.process_way_chunk(chunk_ways)
                    persistence.save_chunk(chunk_index, chunk_segments, (chunk_lat, chunk_lon, chunk_radius))
                    all_merged_segments.extend(chunk_segments)
                else:
                    # Save empty chunk to mark as processed
                    persistence.save_chunk(chunk_index, [], (chunk_lat, chunk_lon, chunk_radius))
                
                processed_chunks.append(chunk_index)
                
                # Save progress
                persistence.save_progress(processed_chunks, total_chunks, metadata)
                
                pbar.set_postfix({
                    'total_segments': len(all_merged_segments),
                    'completed': f"{len(processed_chunks)}/{total_chunks}"
                })
                pbar.update(1)
                
                # Clean up memory
                del chunk_ways
                gc.collect()
                
                # Small delay to be respectful to the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\nError processing chunk {chunk_index}: {e}")
                print("Progress has been saved. You can resume this analysis later.")
                raise
    
    print(f"\nCompleted all chunks! Collected {len(all_merged_segments)} road segments total")
    
    # Now proceed to elevation analysis
    return complete_analysis_from_segments(all_merged_segments, elevation_fetcher,
                                         cross_chunk_merger, metadata, 
                                         min_score, unit_system, persistence)

def process_all_chunks(persistence, road_analyzer, chunk_merger, cross_chunk_merger,
                      elevation_fetcher, chunks, metadata, min_score, unit_system):
    """Process all chunks for a new analysis."""
    
    total_chunks = len(chunks)
    processed_chunks = []
    all_merged_segments = []
    
    print("Processing road data in chunks with checkpoint saving...")
    
    with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for chunk_index, (chunk_lat, chunk_lon, chunk_radius) in enumerate(chunks):
            
            try:
                chunk_ways = road_analyzer.get_roads_in_chunk(chunk_lat, chunk_lon, chunk_radius)
                
                if chunk_ways:
                    chunk_segments = chunk_merger.process_way_chunk(chunk_ways)
                    persistence.save_chunk(chunk_index, chunk_segments, (chunk_lat, chunk_lon, chunk_radius))
                    all_merged_segments.extend(chunk_segments)
                else:
                    # Save empty chunk to mark as processed
                    persistence.save_chunk(chunk_index, [], (chunk_lat, chunk_lon, chunk_radius))
                
                processed_chunks.append(chunk_index)
                
                # Save progress
                persistence.save_progress(processed_chunks, total_chunks, metadata)
                
                pbar.set_postfix({
                    'status': 'processed',
                    'total_segments': len(all_merged_segments)
                })
                pbar.update(1)
                
                # Clean up memory
                del chunk_ways
                gc.collect()
                
                # Small delay to be respectful to the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\nError processing chunk {chunk_index}: {e}")
                print("Progress has been saved. You can resume this analysis later.")
                raise
    
    print(f"\nCollected {len(all_merged_segments)} road segments from all chunks")
    
    # Continue with elevation analysis
    return complete_analysis_from_segments(all_merged_segments, elevation_fetcher,
                                         cross_chunk_merger, metadata, 
                                         min_score, unit_system, persistence)

def analyze_area(address: str, country: str = None, radius_km: float = 15, 
                               surface_filter: str = 'all', unit_system: str = 'imperial',
                               min_score: float = 6000, chunk_size_km: float = 5.0,
                               center_lat: float = None, center_lon: float = None,
                               formatted_address: str = None):
    """Analysis for areas with resumable chunk processing."""
    
    print(f"=== STARTING ANALYSIS ===")
    print(f"Analyzing {surface_filter} roads in {radius_km:.1f}km radius")
    print(f"Using {chunk_size_km}km chunks for memory efficiency")
    
    total_start_time = time.time()
    
    # Create analysis ID
    radius_str = str(radius_km).replace('.', '')
    base_analysis_id = f"{address}_{surface_filter}_{radius_str}km"
    base_analysis_id = "".join(c for c in base_analysis_id if c.isalnum() or c in ('_', '-'))[:40]
    
    # Check for existing incomplete analyses
    base_dir = Path("climb_analysis_checkpoints")
    existing_analysis_id = find_existing_analysis(base_analysis_id, base_dir)
    
    if existing_analysis_id:
        print(f"Found existing analysis: {existing_analysis_id}")
        
        # Check what phase the analysis is in
        persistence = ChunkPersistenceManager(existing_analysis_id)
        processed_chunks, total_chunks, metadata = persistence.load_progress()
        elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()
        
        print(f"Chunk progress: {len(processed_chunks)}/{total_chunks}")
        print(f"Elevation progress: {len(elevation_mapping)} coordinates")
        print(f"Elevation batch info: {elevation_batch_info}")
        
        # Determine phase
        chunks_complete = len(processed_chunks) >= total_chunks
        elevation_in_progress = len(elevation_mapping) > 0
        
        print(f"Chunks complete: {chunks_complete}")
        print(f"Elevation in progress: {elevation_in_progress}")
        
        if chunks_complete and elevation_in_progress:
            print(f"PHASE: ELEVATION FETCHING")
            print(f"  - All chunks completed: {len(processed_chunks)}/{total_chunks}")
            print(f"  - Elevation progress: {len(elevation_mapping)} coordinates")
            
            resume_choice = input("Resume elevation fetching? (y/n): ").strip().lower()
        elif chunks_complete and not elevation_in_progress:
            print(f"PHASE: READY FOR ELEVATION") 
            print(f"  - All chunks completed: {len(processed_chunks)}/{total_chunks}")
            print(f"  - No elevation progress yet")
            
            resume_choice = input("Start elevation analysis? (y/n): ").strip().lower()
        else:
            print(f"PHASE: CHUNK PROCESSING")
            print(f"  - Chunks completed: {len(processed_chunks)}/{total_chunks}")
            
            resume_choice = input("Resume chunk processing? (y/n): ").strip().lower()
        
        if resume_choice == 'y':
            return resume_chunk_processing(existing_analysis_id, min_score, unit_system)
        else:
            print("Session resume declined.")
    
    # Create new analysis
    analysis_id = f"{base_analysis_id}_{int(time.time())}"
    print(f"Starting new analysis: {analysis_id}")
    
    # Initialize components
    road_analyzer = ChunkedRoadNetworkAnalyzer(surface_filter, chunk_size_km)
    elevation_fetcher = ElevationFetcher(batch_size=1000)
    chunk_merger = MemoryEfficientMerger()
    cross_chunk_merger = CrossChunkRoadMerger(coordinate_tolerance=0.002)
    
    # Get center coordinates
    if center_lat is not None and center_lon is not None:
        print(f"Using pre-calculated coordinates for: {formatted_address or address}")
        print(f"Center: {formatted_address or address}")
        print(f"Coordinates: {center_lat:.6f}, {center_lon:.6f}\n")
        final_formatted_address = formatted_address or address
    else:
        center_lat, center_lon, final_formatted_address = road_analyzer.get_coordinates_from_address(address, country)
        print(f"Center: {final_formatted_address}")
        print(f"Coordinates: {center_lat:.6f}, {center_lon:.6f}\n")
    
    # Calculate chunks
    chunks = road_analyzer.calculate_chunks(center_lat, center_lon, radius_km)
    total_chunks = len(chunks)
    
    # Save analysis metadata
    analysis_metadata = {
        'address': address,
        'country': country,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'radius_km': radius_km,
        'surface_filter': surface_filter,
        'unit_system': unit_system,
        'chunk_size_km': chunk_size_km,
        'total_chunks': total_chunks,
        'chunks': chunks,
        'formatted_address': final_formatted_address
    }
    
    # Initialize persistence manager
    persistence = ChunkPersistenceManager(analysis_id)
    
    # Process chunks
    return process_all_chunks(persistence, road_analyzer, chunk_merger, cross_chunk_merger,
                            elevation_fetcher, chunks, analysis_metadata, min_score, unit_system)




def resume_analysis_from_chunks(persistence: ChunkPersistenceManager, metadata: Dict, 
                               min_score: float, unit_system: str):
    """Resume analysis from saved chunks."""
    print("Loading processed chunks from disk...")
    
    # Load all processed chunks
    processed_chunks, total_chunks, _ = persistence.load_progress()
    all_merged_segments = []
    
    with tqdm(total=len(processed_chunks), desc="Loading saved chunks", unit="chunk") as pbar:
        for chunk_index in processed_chunks:
            chunk_data = persistence.load_chunk(chunk_index)
            if chunk_data:
                all_merged_segments.extend(chunk_data)
            pbar.update(1)
    
    print(f"Loaded {len(all_merged_segments)} road segments from {len(processed_chunks)} chunks")
    
    # Initialize remaining components
    elevation_fetcher = ElevationFetcher(batch_size=1000)
    cross_chunk_merger = CrossChunkRoadMerger(coordinate_tolerance=0.002)
    
    return complete_analysis_from_segments(all_merged_segments, elevation_fetcher,
                                         cross_chunk_merger, metadata, 
                                         min_score, unit_system, persistence)

def complete_analysis_from_segments(all_merged_segments: List[Dict], 
                                  elevation_fetcher: ElevationFetcher,
                                  cross_chunk_merger: CrossChunkRoadMerger,
                                  metadata: Dict, min_score: float, 
                                  unit_system: str, persistence: ChunkPersistenceManager):
    """Complete the analysis from merged segments with elevation checkpointing."""
    
    print(f"  - Merged segments: {len(all_merged_segments)}")

    print("Deduplicating segments...")
    all_merged_segments = deduplicate_segments(all_merged_segments)

    print("Performing cross-chunk road merging...")
    all_merged_segments = cross_chunk_merger.merge_cross_chunk_segments(all_merged_segments)
    
    # Extract coordinates
    print("Extracting coordinates for elevation analysis...")
    all_coordinates, coordinate_to_node_ids = extract_coordinates_from_segments(all_merged_segments)
        
    # Check for existing elevation progress
    elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()
    print(f"Found existing elevation progress: {len(elevation_mapping)} coordinates")
    
    # Get elevation data with checkpoint support
    print("\nFetching elevation data with checkpoint support...")
    elevations = elevation_fetcher.fetch_elevations_for_coordinates_with_checkpoints(
        all_coordinates, persistence, "Fetching elevations"
    )
    
    # Create elevation mapping
    node_elevations = {}
    for i, (coord, elevation) in enumerate(zip(all_coordinates, elevations)):
        if elevation is not None:
            for node_id in coordinate_to_node_ids[coord]:
                node_elevations[node_id] = elevation
    
    print(f"Successfully mapped elevations for {len(node_elevations)} nodes")
    
    # Analyze climbs
    print("\nAnalyzing climbs...")
    analyzer = ClimbAnalyzer(metadata.get('surface_filter', 'all'), unit_system)
    analyzer.node_elevations = node_elevations
    
    climbs = analyzer.analyze_merged_roads(all_merged_segments)
    
    # GET ANALYSIS CENTER FROM METADATA
    analysis_center = (metadata.get('center_lat'), metadata.get('center_lon'))
    
    # Print results WITH analysis center for location lookup
    print("\nGenerating results with location lookup...")
    df = analyzer.print_climb_results(min_score, unit_system, analysis_center)
    
    # Return the persistence manager with the results for use in main()
    return climbs, df, persistence

def extract_coordinates_from_segments(segments: List[Dict]) -> Tuple[List[Tuple[float, float]], Dict]:
    """Extract unique coordinates from segments."""
    coordinates = []
    coord_to_node_ids = defaultdict(list)
    
    for segment in segments:
        if 'nodes' in segment:
            for node in segment['nodes']:
                if hasattr(node, 'lat') and hasattr(node, 'lon') and hasattr(node, 'id'):
                    coord = (round(float(node.lat), 6), round(float(node.lon), 6))
                    
                    # Only add if not already seen
                    if coord not in coord_to_node_ids:
                        coordinates.append(coord)
                    
                    coord_to_node_ids[coord].append(node.id)
    
    return coordinates, coord_to_node_ids

class ClimbAnalyzer:
    """Climb analyzer with detailed climb metrics and batch location lookup."""
    
    def __init__(self, surface_filter: str = 'all', unit_system: str = 'imperial'):
        self.surface_filter = surface_filter
        self.unit_system = unit_system
        self.node_elevations = {}
        self.climbs = []
        self.geolocator = Nominatim(user_agent="climb_analyzer")

    def analyze_merged_roads(self, merged_roads: List[Dict]) -> List:
        """Analyze roads."""
        self.climbs = []
        
        print(f"Analyzing {len(merged_roads)} road segments...")
        
        with tqdm(total=len(merged_roads), desc="Analyzing climbs", unit="roads") as pbar:
            for road_segment in merged_roads:
                try:
                    climb_metrics = self.calculate_climb_metrics(road_segment)
                    if climb_metrics:
                        self.climbs.append(climb_metrics)
                    
                    if len(self.climbs) % 100 == 0:
                        pbar.set_postfix({'climbs_found': len(self.climbs)})
                    
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.update(1)
                    continue
        
        print(f"Found {len(self.climbs)} climbs")
        return self.climbs

    def batch_reverse_geocode_climbs(self, climbs: List, analysis_center: Tuple[float, float]) -> List[Dict]:
        """
        Efficiently batch reverse geocode climb starting points and calculate distances.
        """
        if not climbs:
            return []
        
        print("Looking up locations and calculating distances for climbs...")
        
        # Extract unique starting coordinates to minimize reverse geocoding calls
        unique_coords = {}
        climb_to_coord = []
        
        for i, climb in enumerate(climbs):
            # Get starting coordinates from first node
            start_coord = self._get_climb_start_coordinates(climb)
            
            if start_coord:
                # Round coordinates to reduce unique lookups (within ~100m precision)
                rounded_coord = (round(start_coord[0], 3), round(start_coord[1], 3))
                unique_coords[rounded_coord] = start_coord
                climb_to_coord.append(rounded_coord)
            else:
                climb_to_coord.append(None)
        
        # Batch reverse geocode unique coordinates
        coord_to_location = {}
        total_unique = len(unique_coords)
        
        if total_unique > 0:
            print(f"Reverse geocoding {total_unique} unique locations...")
            
            with tqdm(total=total_unique, desc="Looking up locations", unit="coords") as pbar:
                for rounded_coord, actual_coord in unique_coords.items():
                    try:
                        location = self.geolocator.reverse(
                            f"{actual_coord[0]:.6f},{actual_coord[1]:.6f}",
                            timeout=10,
                            exactly_one=True
                        )
                        
                        if location and location.address:
                            # Parse location components
                            address_parts = location.raw.get('address', {})
                            
                            # Extract city (try multiple possible fields)
                            city = (address_parts.get('city') or 
                                   address_parts.get('town') or 
                                   address_parts.get('village') or 
                                   address_parts.get('hamlet') or
                                   address_parts.get('county') or
                                   "Unknown")
                            
                            # Extract state/province
                            state = (address_parts.get('state') or 
                                    address_parts.get('province') or
                                    address_parts.get('region') or
                                    "Unknown")
                            
                            coord_to_location[rounded_coord] = {
                                'city': city,
                                'state': state,
                                'full_address': location.address
                            }
                        else:
                            coord_to_location[rounded_coord] = {
                                'city': 'Unknown',
                                'state': 'Unknown', 
                                'full_address': 'Location not found'
                            }
                            
                    except Exception as e:
                        # Handle geocoding failures gracefully
                        coord_to_location[rounded_coord] = {
                            'city': 'Lookup Failed',
                            'state': 'Lookup Failed',
                            'full_address': f'Error: {str(e)[:50]}'
                        }
                    
                    pbar.update(1)
                    
                    # Be respectful to the geocoding service
                    time.sleep(0.2)  # 200ms delay between requests
        
        # Calculate distances and compile results
        results = []
        for i, climb in enumerate(climbs):
            rounded_coord = climb_to_coord[i]
            
            if rounded_coord and rounded_coord in coord_to_location:
                location_info = coord_to_location[rounded_coord]
                
                # Calculate distance from analysis center to climb start
                actual_coord = unique_coords[rounded_coord]
                distance_km = self.calculate_distance(
                    analysis_center[0], analysis_center[1],
                    actual_coord[0], actual_coord[1]
                )
                
                results.append({
                    'city': location_info['city'],
                    'state': location_info['state'],
                    'distance_km': distance_km,
                    'start_lat': actual_coord[0],
                    'start_lon': actual_coord[1]
                })
            else:
                results.append({
                    'city': 'Unknown',
                    'state': 'Unknown',
                    'distance_km': 0.0,
                    'start_lat': None,
                    'start_lon': None
                })
        
        return results

    def _get_climb_start_coordinates(self, climb) -> Optional[Tuple[float, float]]:
        """Extract starting coordinates for a climb."""
        # Check if climb has direct node access
        if hasattr(climb, 'nodes') and climb.nodes:
            first_node = climb.nodes[0]
            if hasattr(first_node, 'lat') and hasattr(first_node, 'lon'):
                return (float(first_node.lat), float(first_node.lon))
        
        # Alternative: Use OSM API to get way coordinates (less efficient)
        if hasattr(climb, 'way_ids') and climb.way_ids:
            return self._get_way_start_from_osm(climb.way_ids[0])
        
        return None

    def _get_way_start_from_osm(self, way_id: int) -> Optional[Tuple[float, float]]:
        """Get starting coordinates of an OSM way using Overpass API."""
        try:
            # Query for the specific way
            query = f"""
            [out:json][timeout:10];
            way({way_id});
            (._;>;);
            out geom;
            """
            
            if OVERPASS_API_URL:
                response = requests.post(
                    OVERPASS_API_URL,
                    data=query,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=15
                )
            else:
                # Use public Overpass API as fallback
                response = requests.post(
                    "https://overpass-api.de/api/interpreter",
                    data=query,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=15
                )
            
            if response.status_code == 200:
                data = response.json()
                
                # Find the way and get its first node coordinates
                for element in data.get('elements', []):
                    if element.get('type') == 'way' and element.get('id') == way_id:
                        # Check for geometry data first (preferred)
                        if 'geometry' in element and element['geometry']:
                            first_point = element['geometry'][0]
                            if 'lat' in first_point and 'lon' in first_point:
                                return (first_point['lat'], first_point['lon'])
                        
                        # Fallback: get first node ID and find its coordinates
                        if 'nodes' in element and element['nodes']:
                            first_node_id = element['nodes'][0]
                            
                            # Find the node with this ID
                            for node_element in data.get('elements', []):
                                if (node_element.get('type') == 'node' and 
                                    node_element.get('id') == first_node_id):
                                    return (node_element['lat'], node_element['lon'])
        
        except Exception as e:
            print(f"Warning: Could not fetch coordinates for way {way_id}: {e}")
        
        return None

    def calculate_climb_metrics(self, road_segment: Dict):
        """Calculate detailed climb metrics with node storage."""
        street_name = road_segment.get('street_name', 'Unknown')
        nodes = road_segment.get('nodes', [])
        way_ids = road_segment.get('way_ids', [])
        surface = road_segment.get('surface', 'unknown')
        tracktype = road_segment.get('tracktype', 'N/A')
        
        if len(nodes) < 2:
            return None
        
        # Extract coordinates and elevations
        coordinates = []
        elevations = []
        valid_nodes = []
        
        for node in nodes:
            if hasattr(node, 'id') and node.id in self.node_elevations:
                coordinates.append((float(node.lat), float(node.lon)))
                elevations.append(self.node_elevations[node.id])
                valid_nodes.append(node)
        
        if len(elevations) < 2:
            return None
        
        # Calculate distances between consecutive points
        distances = []
        total_distance = 0.0
        
        for i in range(len(coordinates) - 1):
            dist = self.calculate_distance(
                coordinates[i][0], coordinates[i][1],
                coordinates[i+1][0], coordinates[i+1][1]
            )
            distances.append(dist)
            total_distance += dist
        
        if total_distance == 0:
            return None
        
        # Calculate elevation metrics
        min_elevation = min(elevations)
        max_elevation = max(elevations)
        height = max_elevation - min_elevation
        
        # Calculate elevation gain (sum of positive elevation changes)
        elevation_gain = 0.0
        max_grade = 0.0
        
        for i in range(len(elevations) - 1):
            elev_change = elevations[i+1] - elevations[i]
            if elev_change > 0:
                elevation_gain += elev_change
            
            # Calculate grade for this segment
            if distances[i] > 0:
                grade = abs(elev_change / (distances[i] * 1000)) * 100
                max_grade = max(max_grade, grade)
        
        # Calculate average grade
        avg_grade = (height / (total_distance * 1000)) * 100 if total_distance > 0 else 0
        
        # Calculate climb score
        climb_score = total_distance * 1000 * avg_grade if avg_grade > 0 else 0
        
        # Calculate prominence (simplified - height gained from lowest point)
        prominence = height
        
        # Distance between start and end points
        if len(coordinates) >= 2:
            distance_km = self.calculate_distance(
                coordinates[0][0], coordinates[0][1],
                coordinates[-1][0], coordinates[-1][1]
            )
        else:
            distance_km = total_distance
        
        # Determine climb category
        category = self.categorize_climb(avg_grade, total_distance * 1000, elevation_gain)
        
        # Create OSM links
        osm_links = []
        for way_id in way_ids:
            osm_links.append(f"[{way_id}](https://www.openstreetmap.org/way/{way_id})")
        
        return ClimbMetrics(
            street_name=street_name,
            climb_category=category,
            climb_score=climb_score,
            elevation_gain=elevation_gain,
            height=height,
            prominence=prominence,
            length_km=total_distance,
            distance_km=distance_km,
            avg_grade=avg_grade,
            max_grade=max_grade,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            surface=surface,
            tracktype=tracktype,
            tracktype_definition=road_segment.get('tracktype_definition', 'N/A'),
            way_ids=way_ids,
            osm_links=osm_links,
            nodes=nodes  # Store the nodes for location lookup
        )

    def categorize_climb(self, avg_grade: float, length_m: float, elevation_gain_m: float) -> str:
        """Categorize climb based on climb score (cycling categorization system)."""
        # Calculate climb score for categorization
        climb_score = length_m * avg_grade
        
        if climb_score > 80000:
            return "HC"  # Hors Catégorie (beyond categorization)
        elif climb_score > 64000:
            return "1"
        elif climb_score > 32000:
            return "2"
        elif climb_score > 16000:
            return "3"
        elif climb_score > 8000:
            return "4"
        else:
            return "N/A"  # Below category 4 threshold

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def print_climb_results(self, min_score: float = 6000.0, unit_system: str = None, 
                           analysis_center: Tuple[float, float] = None):
        """Print results with location lookup and distance calculation."""
        if not self.climbs:
            print("No climbs found")
            return None
        
        # Use provided unit system or fall back to instance unit system
        units = unit_system or self.unit_system
        
        # Filter climbs by minimum score
        filtered_climbs = [c for c in self.climbs if c.climb_score >= min_score]
        
        if not filtered_climbs:
            print(f"No climbs found with score >= {min_score}")
            return None
        
        # Sort by climb score (descending)
        sorted_climbs = sorted(filtered_climbs, key=lambda x: x.climb_score, reverse=True)
        
        # Batch lookup locations and calculate distances
        location_data = []
        if analysis_center:
            location_data = self.batch_reverse_geocode_climbs(sorted_climbs, analysis_center)
        else:
            # Fill with empty data if no analysis center provided
            location_data = [{'city': 'N/A', 'state': 'N/A', 'distance_km': 0.0} 
                            for _ in sorted_climbs]
        
        # Print header
        header_line = "=" * 160  # Extended for new columns
        print(f"\n{header_line}")
        print(f"CLIMB ANALYSIS RESULTS ({len(sorted_climbs)} climbs) - Surface filter: {self.surface_filter}")
        print(f"{header_line}")
        
        # Prepare data for table
        data = []
        for i, climb in enumerate(sorted_climbs):
            # Convert units based on system
            if units == 'imperial':
                elev_gain = climb.elevation_gain * 3.28084  # m to ft
                height = climb.height * 3.28084  # m to ft  
                prominence = climb.prominence * 3.28084  # m to ft
                length = climb.length_km * 0.621371  # km to mi
                distance = climb.distance_km * 0.621371  # km to mi
                center_distance = location_data[i]['distance_km'] * 0.621371  # km to mi
                elev_unit = "ft"
                dist_unit = "mi"
            else:
                elev_gain = climb.elevation_gain
                height = climb.height
                prominence = climb.prominence
                length = climb.length_km
                distance = climb.distance_km
                center_distance = location_data[i]['distance_km']
                elev_unit = "m"
                dist_unit = "km"
            
            # Format way IDs and OSM links - show only first (start of climb)
            way_ids_str = str(climb.way_ids[0]) if climb.way_ids else "N/A"
            osm_links_str = climb.osm_links[0] if climb.osm_links else "N/A"
            
            row_data = {
                'Street Name': climb.street_name,
                'City': location_data[i]['city'],
                'State': location_data[i]['state'],
                f'From Center ({dist_unit})': round(center_distance, 1),
                'Category': climb.climb_category,
                'Score': int(climb.climb_score),
                f'Elev Gain ({elev_unit})': int(elev_gain),
                f'Height ({elev_unit})': int(height),
                f'Prominence ({elev_unit})': int(prominence),
                f'Length ({dist_unit})': round(length, 2),
                f'Distance ({dist_unit})': round(distance, 2),
                'Avg Grade (%)': round(climb.avg_grade, 2),
                'Max Grade (%)': round(climb.max_grade, 2),
                'Surface': climb.surface,
                'Tracktype': climb.tracktype,
                'Way ID': way_ids_str,
                'OSM Link': osm_links_str
            }
            data.append(row_data)
        
        # Create and print DataFrame
        df = pd.DataFrame(data)
        
        # Print the table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df.to_string(index=False))
        
        # Reset pandas options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width') 
        pd.reset_option('display.max_colwidth')
        
        return df

def get_surface_filter_choice() -> str:
    """Get surface filter choice from user input."""
    print("\nSurface filter options:")
    print("1. Paved roads only (highways, primary, secondary roads)")
    print("2. Gravel roads (tracks with gravel surface)")
    print("3. Dirt trails/tracks (OSM track classification)")
    print("4. All road types")
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            return 'paved'
        elif choice == '2':
            return 'gravel'
        elif choice == '3':
            return 'dirt'
        elif choice == '4':
            return 'all'
        elif not choice:
            print('Defaulting to paved roads.')
            return 'paved'
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def get_unit_system_choice() -> str:
    """Get unit system choice from user input."""
    print("\nUnit system options:")
    print("1. Metric (meters, kilometers)")
    print("2. Imperial (feet, miles)")
    
    while True:
        choice = input("Enter your choice (1-2, default imperial): ").strip() or "2"
        if choice == '1':
            return 'metric'
        elif choice == '2':
            return 'imperial'
        else:
            print("Defaulting to Imperial units.")
            return 'imperial'

def get_analysis_scope_choice() -> Tuple[str, str, Optional[float]]:
    """Get analysis scope choice from user input."""
    print("\nAnalysis scope options:")
    print("1. Address with radius (recommended for most users)")
    print("2. Entire state (WARNING: Very large computational task)")
    print("3. Entire country (WARNING: Extremely large computational task)")
    
    while True:
        choice = input("Enter your choice (1-3, default 1): ").strip() or "1"
        
        if choice == '1':
            return 'address', '', None
        elif choice == '2':
            print("\nWARNING STATE ANALYSIS:")
            print("   • This will analyze an entire state (typically 50,000-100,000+ square miles)")
            print("   • Processing time: Several hours to days depending on state size")
            print("   • Memory usage: High (recommend 16GB+ RAM)")
            print("   • API calls: Thousands of requests to elevation service")
            print("   • Consider starting with a smaller radius first")
            
            confirm = input("\nAre you sure you want to proceed with state analysis? (y/n): ").strip().lower()
            if confirm == 'y':
                state = input("Enter state name (e.g., 'Colorado', 'California'): ").strip()
                return 'state', state, None
            else:
                continue
        elif choice == '3':
            print("\nCOUNTRY ANALYSIS WARNING:")
            print("   • This will analyze an entire country (millions of square miles)")
            print("   • Processing time: Days to weeks")
            print("   • Memory usage: Very high (recommend 32GB+ RAM)")
            print("   • API calls: Tens of thousands of elevation requests")
            print("   • May exceed API rate limits and quotas")
            print("   • Strongly recommend starting with state or regional analysis")
            
            confirm = input("\nAre you ABSOLUTELY sure you want to proceed? (y/n): ").strip().lower()
            if confirm == 'y':
                country = input("Enter country name (e.g., 'United States', 'Canada'): ").strip()
                return 'country', country, None
            else:
                continue
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def calculate_state_bounds(state_name: str, country: str = "United States") -> Tuple[float, float, float]:
    """
    Calculate approximate bounds for a state.
    Returns (center_lat, center_lon, radius_km)
    """
    # Accurate state bounds - covers all major US states and Canadian provinces
    
    state_key = state_name.lower().strip()
    
    # Try exact match first
    if state_key in state_data:
        lat, lon, radius = state_data[state_key]
        print(f"Found exact match for '{state_name}'")
        return lat, lon, radius
    
    # Try partial matches
    for key, value in state_data.items():
        if state_key in key or key in state_key:
            lat, lon, radius = value
            print(f"Found partial match: '{state_name}' -> '{key.title()}'")
            return lat, lon, radius
    
    # State not found
    print(f"Warning: State/Province '{state_name}' not found in database.")
    print("Available options:")
    available_states = sorted(state_data.keys())
    for i in range(0, len(available_states), 4):  # Print 4 per line
        line_states = available_states[i:i+4]
        print("  " + ", ".join(s.title() for s in line_states))
    
    # Default to center of continental US with large radius
    print("Using default coordinates (center of continental US)")
    return (39.0, -98.0, 400)

def calculate_country_bounds(country_name: str) -> Tuple[float, float, float]:
    """
    Calculate approximate bounds for a country.
    Returns (center_lat, center_lon, radius_km)
    """

    country_key = country_name.lower().strip()
    
    # Try exact match first
    if country_key in country_data:
        lat, lon, radius = country_data[country_key]
        print(f"Found exact match for '{country_name}'")
        return lat, lon, radius
    
    # Try partial matches
    for key, value in country_data.items():
        if country_key in key or key in country_key:
            lat, lon, radius = value
            print(f"Found partial match: '{country_name}' -> '{key.title()}'")
            return lat, lon, radius
    
    # Country not found
    print(f"Warning: Country '{country_name}' not found in database.")
    print("Available countries (first 20):")
    available_countries = sorted(country_data.keys())
    for i in range(0, min(20, len(available_countries)), 4):  # Print 4 per line, max 20
        line_countries = available_countries[i:i+4]
        print("  " + ", ".join(c.title() for c in line_countries))
    if len(available_countries) > 20:
        print(f"  ... and {len(available_countries) - 20} more")
    
    # Default global center
    print("Using default coordinates (global center)")
    return (0.0, 0.0, 1000)

def deduplicate_segments(segments: List[Dict]) -> List[Dict]:
    """Deduplication by way IDs and coordinate similarity."""
    
    # Step 1: Remove exact way ID duplicates
    seen_way_sets = set()
    deduplicated = []
    
    for segment in segments:
        way_ids = tuple(sorted(segment.get('way_ids', [])))
        if way_ids and way_ids not in seen_way_sets:
            seen_way_sets.add(way_ids)
            deduplicated.append(segment)
        elif not way_ids:  # Keep segments without way_ids
            deduplicated.append(segment)
    
    # Step 2: Remove segments with very similar start/end coordinates (likely duplicates)
    final_segments = []
    for i, segment in enumerate(deduplicated):
        is_duplicate = False
        nodes = segment.get('nodes', [])
        
        if not nodes:
            final_segments.append(segment)
            continue
            
        start_coord = extract_node_coordinates(nodes[0])
        end_coord = extract_node_coordinates(nodes[-1])
        
        if not start_coord or not end_coord:
            final_segments.append(segment)
            continue
        
        # Check against all previous segments
        for j, prev_segment in enumerate(final_segments):
            prev_nodes = prev_segment.get('nodes', [])
            if not prev_nodes:
                continue
                
            prev_start = extract_node_coordinates(prev_nodes[0])
            prev_end = extract_node_coordinates(prev_nodes[-1])
            
            if not prev_start or not prev_end:
                continue
            
            # Check if coordinates are very similar (within 10m)
            start_dist = calculate_distance_km(start_coord[0], start_coord[1], 
                                             prev_start[0], prev_start[1]) * 1000
            end_dist = calculate_distance_km(end_coord[0], end_coord[1], 
                                           prev_end[0], prev_end[1]) * 1000
            
            if start_dist < 10 and end_dist < 10:
                # Very similar segment - likely duplicate
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_segments.append(segment)
    
    print(f"Deduplication: {len(segments)} -> {len(final_segments)} segments")
    return final_segments


def extract_node_coordinates(node):
    """Extract coordinates from a node object."""
    try:
        if hasattr(node, 'lat') and hasattr(node, 'lon'):
            lat = float(node.lat)
            lon = float(node.lon)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
    except (ValueError, TypeError, AttributeError):
        pass
    return None

def analyze_segment_connectivity(seg1: Dict, seg2: Dict, seg1_num: int, seg2_num: int):
    """Analyze if and how two segments can be connected."""
    nodes1 = seg1.get('nodes', [])
    nodes2 = seg2.get('nodes', [])
    
    if not nodes1 or not nodes2:
        print(f"  ERROR: One or both segments have no nodes")
        return
    
    # Get all endpoints
    endpoints1 = {
        'start': extract_node_coordinates(nodes1[0]),
        'end': extract_node_coordinates(nodes1[-1])
    }
    endpoints2 = {
        'start': extract_node_coordinates(nodes2[0]),
        'end': extract_node_coordinates(nodes2[-1])
    }
    
    # Check all possible connections
    connections = []
    for name1, coord1 in endpoints1.items():
        for name2, coord2 in endpoints2.items():
            if coord1 and coord2:
                distance_km = calculate_distance_km(coord1[0], coord1[1], coord2[0], coord2[1])
                distance_m = distance_km * 1000
                connections.append((f"seg{seg1_num}_{name1}", f"seg{seg2_num}_{name2}", distance_m))
    
    # Sort by distance
    connections.sort(key=lambda x: x[2])
    
    print(f"  Connection distances:")
    for conn1, conn2, dist_m in connections:
        status = "CONNECTABLE" if dist_m < 100 else "TOO FAR" if dist_m < 1000 else "VERY FAR"
        print(f"    {conn1} <-> {conn2}: {dist_m:.1f}m ({status})")
    
    # Check for potential merging issues
    closest_distance = connections[0][2] if connections else float('inf')
    
    if closest_distance < 10:
        print(f"  ✓ SHOULD MERGE: Closest endpoints are {closest_distance:.1f}m apart")
    elif closest_distance < 100:
        print(f"  ? MIGHT MERGE: Closest endpoints are {closest_distance:.1f}m apart")
    elif closest_distance < 1000:
        print(f"  ✗ GAP TOO LARGE: Closest endpoints are {closest_distance:.1f}m apart")
    else:
        print(f"  ✗ SEGMENTS VERY FAR: Closest endpoints are {closest_distance:.1f}m apart")
    
    # Check current merger tolerances
    current_tolerance_degrees = 0.0001  # From your current code
    current_tolerance_meters = current_tolerance_degrees * 111000  # Rough conversion
    print(f"  Current merger tolerance: {current_tolerance_meters:.1f}m (may be too strict)")

def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance using Haversine formula (in km)."""
    import math
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon/2) * math.sin(delta_lon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def main():
    """Main function to run the climb analyzer."""

    print("======================")
    print("=== CLIMB ANALYZER ===")
    print("======================")    
    print("Efficiently analyzes road and trail climbs from OpenStretMap.")

    total_start_time = time.time()

    # Get surface filter preference
    surface_filter = get_surface_filter_choice()
    print(f"Selected surface filter: {surface_filter}")

    # Get unit system preference
    unit_system = get_unit_system_choice()
    print(f"Selected unit system: {unit_system}")

    # Get analysis scope
    scope_type, location, radius_km = get_analysis_scope_choice()
    
    if scope_type == 'address':
        # Get address from user
        address = input("\nEnter the street address to analyze, *without* the country code: ").strip()
        if not address:
            print("Please enter a valid address.")
            return

        # Optional country specification for better geocoding accuracy
        country = input("Enter country name (optional, helps with geocoding accuracy): ").strip()
        if not country:
            country = None

        # Get search radius
        try:
            radius_unit = 'miles'
            if unit_system == 'metric':
                radius_unit = 'kilometers'
            radius_input = input(f"Enter the search radius (default 15 {radius_unit}): ").strip() or "15"
            radius = float(radius_input)
            
            # Convert to km for internal processing
            if radius_unit == 'miles':
                radius_km = radius * 1.60934
            else:
                radius_km = radius
        except ValueError:
            radius = 15
            radius_km = 15 * 1.60934 if radius_unit == 'miles' else 15

    elif scope_type == 'state':
        # State analysis - pre-calculate coordinates
        try:
            center_lat, center_lon, radius_km = calculate_state_bounds(location)
            address = f"{location} State Analysis"  # Don't geocode this
            country = "United States"
            formatted_address = f"{location}, United States"
            print(f"State analysis: {location}")
            print(f"Calculated center: {center_lat:.4f}, {center_lon:.4f}")
            print(f"Estimated coverage: {radius_km*2:.0f}km diameter")
        except ValueError as e:
            print(f"Error: {e}")
            return
        
    elif scope_type == 'country':
        # Country analysis  
        try:
            center_lat, center_lon, radius_km = calculate_country_bounds(location)
            address = location
            country = None
            print(f"Country analysis: {location}")
            print(f"Estimated coverage: {radius_km*2:.0f}km diameter")
        except ValueError as e:
            print(f"Error: {e}")
            return

    # Get minimum score threshold
    try:
        min_score = float(input("Enter minimum climb score to display (default 6000): ").strip() or "6000")
    except ValueError:
        min_score = 6000

    # Get chunk size for processing
    chunk_size_km = 8.0  # Default
    if radius_km > 50:  # For large areas, offer chunk size configuration
        print(f"\nLarge area detected ({radius_km:.1f}km radius)")
        try:
            chunk_input = input(f"Enter chunk size in km for processing (default 8.0, smaller = less memory): ").strip()
            if chunk_input:
                chunk_size_km = float(chunk_input)
        except ValueError:
            chunk_size_km = 8.0

    # Display analysis summary
    print(f"\nStarting analysis:")
    print(f"Location: {address}")
    if country:
        print(f"Country: {country}")
    print(f"Surface filter: {surface_filter}")
    print(f"Search radius: {radius_km:.1f} km")
    print(f"Minimum climb score: {min_score}")
    print(f"Chunk size: {chunk_size_km} km")
    
    # Estimate processing time
    estimated_chunks = (radius_km / chunk_size_km) ** 2 * math.pi
    estimated_time_minutes = estimated_chunks * 0.5  # Rough estimate: 30 seconds per chunk
    
    if estimated_time_minutes > 60:
        print(f"WARNING: Estimated processing time: {estimated_time_minutes/60:.1f} hours")
        print(f"   ({estimated_chunks:.0f} chunks to process)")
        
        if estimated_time_minutes > 1440:  # > 24 hours
            print("   This is a very large analysis that may take over a day.")
            confirm = input("   Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Analysis cancelled.")
                return
    else:
        print(f"Estimated processing time: {estimated_time_minutes:.1f} minutes")

    # Run analysis
    try:
        
        climbs, df, persistence = analyze_area(
            address=address,
            country=country,
            radius_km=radius_km,
            surface_filter=surface_filter,
            unit_system=unit_system,
            min_score=min_score,
            chunk_size_km=chunk_size_km,
            center_lat=center_lat if scope_type != 'address' else None,
            center_lon=center_lon if scope_type != 'address' else None,
            formatted_address=formatted_address if scope_type != 'address' else None
            # Remove: enable_location_lookup=enable_location_lookup
        )

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return

    total_duration = time.time() - total_start_time
    print(f"\nTOTAL PROCESSING TIME: {total_duration:.2f} seconds")
    
    # Optional: Save results to XLSX (MOVED BEFORE CLEANUP)
    save_xlsx = input("\nSave results to XLSX file? (y/n): ").strip().lower()
    if save_xlsx == 'y' and df is not None and len(df) > 0:
        # Create a safe filename
        if scope_type == 'address':
            safe_name = "".join(c for c in address if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')[:50]
        else:
            safe_name = location.replace(' ', '_')
            
        filename = f"climbs_{safe_name}_{surface_filter}_{scope_type}_{radius_km:.0f}km.xlsx"
        
        # Save with Excel filters on header row
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Climbs', index=False)
            
            # Add autofilter to header row
            worksheet = writer.sheets['Climbs']
            worksheet.auto_filter.ref = worksheet.dimensions
            
        print(f"Results saved to {filename} (with Excel filters)")
    elif save_xlsx == 'y':
        print("No results to save (no climbs found above minimum score)")

    # NOW HANDLE CLEANUP (MOVED TO AFTER XLS SAVING)
    print("\nCleanup options:")
    print("1. Clean up this analysis only")
    print("2. Clean up all checkpoint directories") 
    print("3. Keep checkpoint files")
    
    cleanup_choice = input("Enter your choice (1-3): ").strip().lower()
    if cleanup_choice == '1':
        persistence.cleanup()
    elif cleanup_choice == '2':
        # Clean up all checkpoint directories
        try:
            import shutil
            base_dir = Path("climb_analysis_checkpoints")
            if base_dir.exists():
                shutil.rmtree(base_dir)
                print("Cleaned up all checkpoint directories")
        except Exception as e:
            print(f"Error cleaning up all checkpoints: {e}")
    else:
        print(f"Checkpoint files preserved in: {persistence.analysis_dir}")

    return climbs

def example_usage():
    """Example of how to use the analyzer programmatically"""
    
    # Example 1: Large city analysis (30-mile radius around Atlanta) with location lookup disabled for speed
    climbs1, df1, persistence1 = analyze_large_area(
        address="Atlanta, Georgia",
        country="US", 
        radius_km=48.3,  # 30 miles
        surface_filter='all',
        unit_system='imperial',
        min_score=3000,
        chunk_size_km=8.0,
        enable_location_lookup=False  # Disable for faster analysis
    )
    
    # Example 2: State analysis (entire Colorado) with location lookup enabled
    climbs2, df2, persistence2 = analyze_large_area(
        address="Colorado, United States",
        country="United States",
        radius_km=250,  # Covers most of Colorado
        surface_filter='paved',
        unit_system='imperial', 
        min_score=5000,
        chunk_size_km=10.0,
        enable_location_lookup=True  # Enable for detailed location info
    )
    
    return climbs1, climbs2

if __name__ == "__main__":
    # Required packages installation note
    print("Required packages: requests, geopy, overpy, numpy, pandas, tqdm, openpyxl")
    print("Install with: pip install requests geopy overpy numpy pandas tqdm openpyxl")
    print("Install with: pip install geopandas")
    print()
    
    main()