#!/usr/bin/env python3
"""
Error logging and reporting module for Climb Analyzer.

Handles:
- Elevation fetch error reporting
- Log file rotation (keep last 100 runs)
- Error statistics tracking
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class ErrorLogger:
    """
    Handles error logging and rotation for climb analyzer.

    Tracks elevation fetch failures and maintains rotating error logs.
    """

    def __init__(self, error_log_path: Path = Path("error.log"), elevation_error_log: Path = Path("elevation_errors.log"), max_runs: int = 100, streaming: bool = True, region_name: str = "Unknown", output_dir: Path = None, base_filename: str = None, successful_datasets: list = None, surface_filter: str = None, min_score: float = None, score_type: str = None, cycling_allowed: bool = None, app_version: str = None):
        """
        Initialize error logger.

        Args:
            error_log_path: Path to error log file
            elevation_error_log: Path to detailed elevation error log
            max_runs: Maximum number of runs to keep in log (default: 100)
            streaming: If True, write errors to file immediately (default: True)
            region_name: Name of the region being analyzed (for CSV header)
            output_dir: Directory for output files (defaults to "./output")
            base_filename: Base filename to use for elevation error log (without extension)
            successful_datasets: List of datasets successfully used (not removed due to unavailability)
            surface_filter: Surface filter used (e.g., 'all', 'paved', 'gravel,dirt')
            min_score: Minimum score threshold used
            score_type: Score type used (e.g., 'basic', 'fiets', 'pdi')
            cycling_allowed: Whether cycling filter was enabled
            app_version: Version of the Climb Analyzer application
        """
        # If output_dir and base_filename are provided, construct elevation error log path
        if output_dir and base_filename:
            output_dir = Path(output_dir)
            # base_filename already includes "_errors_" so just add .txt extension (CSV format inside)
            self.elevation_error_log = output_dir / f"{base_filename}.txt"
            self.error_log_path = output_dir / "error.log"
        else:
            self.error_log_path = Path(error_log_path)
            self.elevation_error_log = Path(elevation_error_log)

        self.max_runs = max_runs
        self.current_run_errors = {}
        self.elevation_errors = []  # Collect errors during run
        self.streaming = streaming
        self.elevation_log_file = None
        self.error_count = 0
        self.region_name = region_name
        self.run_start_time = None
        self.successful_datasets = successful_datasets or []

        # Filter parameters for CSV header
        self.surface_filter = surface_filter
        self.min_score = min_score
        self.score_type = score_type
        self.cycling_allowed = cycling_allowed
        self.app_version = app_version or "unknown"

        # Analysis result statistics (set later after analysis completes)
        self.climb_count = None
        self.total_coords_in_ways = None
        self.failed_coords = None

    def set_analysis_stats(self, climb_count: int = None, total_coords: int = None, failed_coords: int = None):
        """
        Set analysis statistics for inclusion in CSV footer.

        Args:
            climb_count: Total number of climbs found in the analysis
            total_coords: Total number of coordinates in analyzed ways
            failed_coords: Number of coordinates that failed elevation fetch
        """
        if climb_count is not None:
            self.climb_count = climb_count
        if total_coords is not None:
            self.total_coords_in_ways = total_coords
        if failed_coords is not None:
            self.failed_coords = failed_coords

    def set_successful_datasets(self, datasets: list):
        """
        Set the list of successful datasets used for elevation fetching.

        Args:
            datasets: List of dataset names that were successfully used
        """
        self.successful_datasets = datasets or []

    def start_elevation_logging(self, resume_from_checkpoint: bool = False, checkpoint_error_count: int = 0):
        """
        Start streaming elevation errors to file.

        Args:
            resume_from_checkpoint: If True, append to existing log file instead of overwriting
            checkpoint_error_count: Error count to restore when resuming from checkpoint
        """
        if self.streaming:
            try:
                self.run_start_time = datetime.now()

                # Check if we should resume from existing log file
                if resume_from_checkpoint and self.elevation_error_log.exists():
                    # Append to existing log file
                    self.elevation_log_file = open(self.elevation_error_log, "a")
                    self.error_count = checkpoint_error_count

                    # Add a resume marker comment
                    self.elevation_log_file.write(f"# --- Resumed at: {self.run_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    self.elevation_log_file.flush()
                    print(f"  Resuming error log with {self.error_count:,} existing entries")
                else:
                    # Start fresh - create new log file
                    self.elevation_log_file = open(self.elevation_error_log, "w")

                    # Write metadata as CSV comments (lines starting with #)
                    self.elevation_log_file.write("# Elevation Fetch Error Log\n")
                    self.elevation_log_file.write(f"# Climb Analyzer Version: {self.app_version}\n")
                    self.elevation_log_file.write(f"# Region: {self.region_name}\n")
                    self.elevation_log_file.write(f"# Analysis Started: {self.run_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                    # Add filter parameters used for the analysis
                    if self.surface_filter is not None:
                        self.elevation_log_file.write(f"# Surface Filter: {self.surface_filter}\n")
                    if self.min_score is not None:
                        self.elevation_log_file.write(f"# Minimum Score: {self.min_score}\n")
                    if self.score_type is not None:
                        self.elevation_log_file.write(f"# Score Type: {self.score_type}\n")
                    if self.cycling_allowed is not None:
                        cycling_status = "enabled" if self.cycling_allowed else "disabled"
                        self.elevation_log_file.write(f"# Cycling Filter: {cycling_status}\n")

                    # Add successful datasets information (numbered by priority order)
                    if self.successful_datasets:
                        datasets_str = ", ".join(f"{i}. {ds}" for i, ds in enumerate(self.successful_datasets, 1))
                        self.elevation_log_file.write(f"# Elevation Data Sources (priority order): {datasets_str}\n")
                    else:
                        self.elevation_log_file.write("# Elevation Data Sources: Unknown\n")

                    self.elevation_log_file.write("# Format: CSV with pipe-separated (|) datasets in datasets_tried column\n")
                    self.elevation_log_file.write("#\n")

                    # Write CSV header
                    self.elevation_log_file.write("level,timestamp,latitude,longitude,street_name,osm_way_id,datasets_tried,primary_dataset,successful_dataset,message\n")
                    self.elevation_log_file.flush()
                    self.error_count = 0
            except Exception as e:
                print(f"Warning: Could not open elevation error log for streaming: {e}")
                self.streaming = False

    def stop_elevation_logging(self, output_file_count: int = 1):
        """
        Stop streaming and reorganize log file with summary at top.

        Args:
            output_file_count: Number of output xlsx files created (for split files)
        """
        if self.elevation_log_file:
            try:
                # Close the file first
                self.elevation_log_file.close()
                self.elevation_log_file = None

                # Calculate summary statistics
                run_end_time = datetime.now()
                duration = (run_end_time - self.run_start_time).total_seconds() if self.run_start_time else 0

                # Read the entire file content (header + CSV data)
                with open(self.elevation_error_log) as f:
                    lines = f.readlines()

                # Separate header comments from CSV data
                header_lines = []
                csv_header = None
                csv_data_lines = []

                in_header = True
                for line in lines:
                    if in_header and line.startswith('#'):
                        # Update the datasets line if we now have successful datasets
                        if "# Elevation Data Sources:" in line and self.successful_datasets:
                            datasets_str = ", ".join(f"{i}. {ds}" for i, ds in enumerate(self.successful_datasets, 1))
                            line = f"# Elevation Data Sources (priority order): {datasets_str}\n"
                        header_lines.append(line)
                    elif in_header and not line.startswith('#') and line.strip():
                        # This is the CSV header row
                        csv_header = line
                        in_header = False
                    elif not in_header:
                        csv_data_lines.append(line)

                # Build summary section
                summary_lines = []
                summary_lines.append("#\n")
                summary_lines.append("# ==================== ANALYSIS SUMMARY ====================\n")
                summary_lines.append(f"# Analysis Completed: {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                summary_lines.append(f"# Duration: {duration:.1f} seconds\n")
                summary_lines.append("#\n")

                # Output files created
                if output_file_count > 1:
                    summary_lines.append(f"# Output Files Created: {output_file_count} (split files)\n")
                else:
                    summary_lines.append(f"# Output Files Created: {output_file_count}\n")

                # Climb results
                if self.climb_count is not None:
                    summary_lines.append(f"# Total Climbs Found: {self.climb_count:,}\n")

                # Elevation error statistics
                summary_lines.append(f"# Total Error Entries (Coordinates): {self.error_count:,}\n")

                if self.total_coords_in_ways is not None and self.failed_coords is not None:
                    if self.total_coords_in_ways > 0:
                        error_pct = (self.failed_coords / self.total_coords_in_ways) * 100
                        summary_lines.append(f"# Total Coordinates Analyzed: {self.total_coords_in_ways:,}\n")
                        summary_lines.append(f"# Failed Elevation Fetches: {self.failed_coords:,}\n")
                        summary_lines.append(f"# Elevation Error Percentage: {error_pct:.2f}%\n")

                summary_lines.append("# ==========================================================\n")
                summary_lines.append("#\n")

                # Rewrite file: header -> summary -> CSV header -> CSV data
                with open(self.elevation_error_log, 'w') as f:
                    # Write initial header
                    f.writelines(header_lines)

                    # Write summary section
                    f.writelines(summary_lines)

                    # Write CSV header
                    if csv_header:
                        f.write(csv_header)

                    # Write CSV data
                    f.writelines(csv_data_lines)

                print(f"✓ Elevation error log saved to {self.elevation_error_log} ({self.error_count:,} coordinates)")
            except Exception as e:
                print(f"Warning: Error closing elevation error log: {e}")

    def log_coordinate_failure(
        self,
        coordinate: tuple,
        street_name: str = "Unknown",
        osm_way_id: str = "Unknown",
        datasets_tried: Optional[list] = None,
        primary_dataset: str = "Unknown",
        successful_dataset: Optional[str] = None,
        level: str = "ERROR"
    ):
        """
        Log a single coordinate elevation fetch result (failure or fallback success).

        Args:
            coordinate: (lat, lon) tuple
            street_name: Name of the street/way
            osm_way_id: OSM way ID
            datasets_tried: List of dataset names attempted
            primary_dataset: Primary dataset name (e.g., 'ned10m')
            successful_dataset: Dataset that successfully returned elevation (None if all failed)
            level: 'INFO' for fallback success, 'ERROR' for complete failure
        """
        error_data = {
            "coordinate": coordinate,
            "street_name": street_name,
            "osm_way_id": osm_way_id,
            "datasets_tried": datasets_tried or [],
            "primary_dataset": primary_dataset,
            "successful_dataset": successful_dataset,
            "level": level
        }

        # MEMORY FIX: Only collect in memory if NOT streaming (prevents accumulating millions of errors)
        # In streaming mode, data is written immediately to file and doesn't need to be kept in RAM
        if not self.streaming:
            self.elevation_errors.append(error_data)

        # If streaming, write immediately to CSV
        if self.streaming and self.elevation_log_file:
            try:
                self.error_count += 1
                lat, lon = coordinate

                # Escape fields that might contain commas
                def escape_csv(value):
                    if value is None:
                        return ""
                    value_str = str(value)
                    if ',' in value_str or '"' in value_str or '\n' in value_str:
                        return '"' + value_str.replace('"', '""') + '"'
                    return value_str

                # Determine message based on level
                if level == "INFO":
                    message = f"Primary {primary_dataset} failed, succeeded with {successful_dataset}"
                else:  # ERROR
                    datasets_str = '|'.join(datasets_tried) if datasets_tried else 'None'
                    message = f"All datasets failed: {datasets_str}"

                # Write CSV row
                datasets_tried_str = '|'.join(datasets_tried) if datasets_tried else ''
                row = [
                    level,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{lat:.6f}",
                    f"{lon:.6f}",
                    escape_csv(street_name),
                    escape_csv(osm_way_id),
                    datasets_tried_str,
                    escape_csv(primary_dataset),
                    escape_csv(successful_dataset) if successful_dataset else "",
                    escape_csv(message)
                ]

                self.elevation_log_file.write(','.join(row) + '\n')

                # Flush every 100 errors to balance performance and real-time visibility
                if self.error_count % 100 == 0:
                    self.elevation_log_file.flush()
            except Exception:
                # Silent failure - don't disrupt the main process
                pass

    def write_elevation_errors_to_file(self):
        """Write collected elevation errors to file."""
        if not self.elevation_errors:
            return

        try:
            with open(self.elevation_error_log, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("ELEVATION FETCH FAILURES - DETAILED LOG\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Failures: {len(self.elevation_errors):,}\n")
                f.write("=" * 80 + "\n\n")

                for i, error in enumerate(self.elevation_errors[:1000], 1):  # Limit to first 1000
                    f.write(f"Failure #{i}\n")
                    f.write(f"  Coordinate: {error['coordinate']}\n")
                    f.write(f"  Street/Way: {error['street_name']}\n")
                    f.write(f"  OSM Way ID: {error['osm_way_id']}\n")
                    f.write(f"  Datasets Tried: {', '.join(error['datasets_tried']) if error['datasets_tried'] else 'None'}\n")
                    f.write(f"  Primary Dataset: {error.get('primary_dataset', 'Unknown')}\n")
                    if error.get('successful_dataset'):
                        f.write(f"  Fallback Success: {error['successful_dataset']}\n")
                    f.write("\n")

                if len(self.elevation_errors) > 1000:
                    f.write(f"\n... and {len(self.elevation_errors) - 1000:,} more failures\n")

        except Exception as e:
            print(f"Warning: Could not write elevation errors to file: {e}")

    def log_elevation_errors(
        self,
        report_filename: str,
        total_coords_in_ways: int,
        failed_coords: int,
        total_coords_in_country: int,
        way_percentage: float,
        country_percentage: float,
        way_failures: Optional[Dict] = None,
        timestamp: Optional[str] = None,
    ):
        """
        Log elevation fetch errors to error.log.

        Args:
            report_filename: Name of the report file this relates to
            total_coords_in_ways: Total number of coordinates in all ways analyzed
            failed_coords: Number of coordinates that failed to fetch elevation
            total_coords_in_country: Total coordinates in entire country/region
            way_percentage: Percentage of way coordinates that failed
            country_percentage: Percentage of country coordinates that failed
            way_failures: Optional per-way failure details
            timestamp: Timestamp of run (auto-generated if None)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Read existing log entries
        entries = self._read_log_entries()

        # Create new entry
        new_entry = {
            "timestamp": timestamp,
            "app_version": self.app_version,
            "report_filename": report_filename,
            "total_coords_in_ways": total_coords_in_ways,
            "failed_coords": failed_coords,
            "total_coords_in_country": total_coords_in_country,
            "way_percentage": way_percentage,
            "country_percentage": country_percentage,
            "way_failures": way_failures or {},
        }

        entries.append(new_entry)

        # Keep only last max_runs entries
        if len(entries) > self.max_runs:
            entries = entries[-self.max_runs :]

        # Write back to file
        self._write_log_entries(entries)

    def _read_log_entries(self) -> list:
        """Read existing log entries from error.log."""
        entries = []

        if not self.error_log_path.exists():
            return entries

        try:
            with open(self.error_log_path) as f:
                current_entry = {}
                for line in f:
                    line = line.strip()

                    # Skip separator lines
                    if line.startswith("=") or line.startswith("-") or not line:
                        if current_entry and "timestamp" in current_entry:
                            entries.append(current_entry)
                            current_entry = {}
                        continue

                    # Parse key-value pairs
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower().replace(" ", "_")
                        value = value.strip()

                        # Convert numeric values
                        try:
                            if "." in value and "%" not in value:
                                current_entry[key] = float(value)
                            elif value.isdigit():
                                current_entry[key] = int(value)
                            else:
                                current_entry[key] = value.replace("%", "")
                        except (ValueError, AttributeError):
                            current_entry[key] = value

                # Don't forget last entry
                if current_entry and "timestamp" in current_entry:
                    entries.append(current_entry)

        except Exception as e:
            print(f"Warning: Could not read error log: {e}")

        return entries

    def _write_log_entries(self, entries: list):
        """Write log entries to error.log."""
        try:
            with open(self.error_log_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("CLIMB ANALYZER - ELEVATION FETCH ERROR LOG\n")
                f.write(f"Last {len(entries)} runs (max: {self.max_runs})\n")
                f.write("=" * 80 + "\n\n")

                for entry in entries:
                    f.write("-" * 80 + "\n")
                    f.write(f"Timestamp: {entry.get('timestamp', 'Unknown')}\n")
                    f.write(f"App Version: {entry.get('app_version', 'unknown')}\n")
                    f.write(f"Report Filename: {entry.get('report_filename', 'Unknown')}\n")
                    f.write(
                        f"Total Coordinates in Ways: {entry.get('total_coords_in_ways', 0):,}\n"
                    )
                    f.write(f"Failed Coordinates: {entry.get('failed_coords', 0):,}\n")
                    f.write(
                        f"Total Coordinates in Country: {entry.get('total_coords_in_country', 0):,}\n"
                    )
                    f.write(f"Failed % of Way Coordinates: {entry.get('way_percentage', 0):.2f}%\n")
                    f.write(
                        f"Failed % of Country Coordinates: {entry.get('country_percentage', 0):.4f}%\n"
                    )

                    # Write per-way failures if available
                    way_failures = entry.get("way_failures", {})
                    if way_failures:
                        f.write(f"\nFailed Ways ({len(way_failures)} total):\n")
                        for way_id, data in list(way_failures.items())[:100]:  # Limit to first 100
                            way_name = data.get("name", f"Way {way_id}")
                            total = data.get("total_coords", 0)
                            failed = data.get("failed_coords", 0)
                            pct = data.get("failure_percentage", 0)
                            f.write(f"  - {way_name}: {failed}/{total} failed ({pct:.2f}%)\n")

                        if len(way_failures) > 100:
                            f.write(f"  ... and {len(way_failures) - 100} more ways\n")

                    f.write("\n")

        except Exception as e:
            print(f"Error writing to error log: {e}")

    def print_error_report(
        self,
        total_coords_in_ways: int,
        failed_coords: int,
        total_coords_in_country: int,
        way_failures: Optional[Dict] = None,
    ):
        """
        Print elevation fetch error report to console.

        Args:
            total_coords_in_ways: Total coordinates in analyzed ways
            failed_coords: Number that failed to fetch
            total_coords_in_country: Total coordinates in country/region
            way_failures: Optional per-way failure details
        """
        print("\n" + "=" * 80)
        print("ELEVATION FETCH ERROR REPORT")
        print("=" * 80)

        # Calculate percentages
        way_percentage = (
            (failed_coords / total_coords_in_ways * 100) if total_coords_in_ways > 0 else 0
        )
        country_percentage = (
            (failed_coords / total_coords_in_country * 100) if total_coords_in_country > 0 else 0
        )

        print(f"Total Coordinates in Analyzed Ways:  {total_coords_in_ways:,}")
        print(f"Coordinates with Failed Elevation:   {failed_coords:,}")
        print(f"Total Coordinates in Country/Region: {total_coords_in_country:,}")
        print()
        print(f"Overall Failed Rate (All Ways):      {way_percentage:.2f}%")
        print(f"Overall Failed Rate (Country):       {country_percentage:.4f}%")

        # Per-way failure details
        if way_failures and len(way_failures) > 0:
            print("\n" + "-" * 80)
            print("FAILURES BY WAY (Merged Ways)")
            print("-" * 80)

            # Show top failures (up to 20 ways)
            shown = 0
            max_show = 20

            for way_id, data in way_failures.items():
                if shown >= max_show:
                    remaining = len(way_failures) - shown
                    print(
                        f"\n... and {remaining} more way(s) with failures (see error.log for complete list)"
                    )
                    break

                way_name = data["name"] or f"Way {way_id}"
                total = data["total_coords"]
                failed = data["failed_coords"]
                pct = data["failure_percentage"]

                # Format way name (truncate if too long)
                display_name = way_name[:50] + "..." if len(way_name) > 50 else way_name

                print(f"\n{display_name}")
                print(f"  Failed: {failed:,} of {total:,} coordinates ({pct:.2f}%)")

                # Show first few failed indices if available
                if "failed_indices" in data and len(data["failed_indices"]) > 0:
                    indices = data["failed_indices"][:5]  # Show first 5
                    indices_str = ", ".join(str(i) for i in indices)
                    if len(data["failed_indices"]) > 5:
                        indices_str += f" ... +{len(data['failed_indices']) - 5} more"
                    print(f"  Failed at coordinate indices: {indices_str}")

                shown += 1

        # Add interpretation
        print("\n" + "-" * 80)
        if way_percentage > 10:
            print("⚠️  WARNING: High elevation fetch failure rate!")
            print("   This may impact analysis quality. Consider:")
            print("   - Checking elevation API status")
            print("   - Reducing ELEVATION_MAX_CONCURRENT in config.yaml")
            print("   - Running analysis again to retry failed coordinates")
        elif way_percentage > 5:
            print("   Moderate elevation fetch failure rate.")
            print("   Results should still be reliable, but consider re-running if critical.")
        else:
            print("✓ Elevation fetch success rate is good.")

        print("=" * 80)

        return way_percentage, country_percentage


class LogRotator:
    """
    Handles log file rotation, keeping last N runs.
    """

    def __init__(self, log_path: Path = Path("climb_analyzer.log"), max_runs: int = 100):
        """
        Initialize log rotator.

        Args:
            log_path: Path to log file
            max_runs: Maximum number of runs to keep (default: 100)
        """
        self.log_path = Path(log_path)
        self.max_runs = max_runs

    def rotate_if_needed(self):
        """
        Rotate log file if it contains more than max_runs entries.

        Looks for run separators and keeps only the most recent max_runs.
        """
        if not self.log_path.exists():
            return

        try:
            # Check file size first
            file_size_mb = self.log_path.stat().st_size / (1024 * 1024)

            # For very large files (>100MB), use tail-based truncation instead of run-based rotation
            # This is much faster and avoids loading huge files into memory
            if file_size_mb > 100:
                print(f"⚠️  Log file is {file_size_mb:.1f}MB - using tail-based truncation")
                import subprocess
                import tempfile

                try:
                    # Keep last 50MB (~500K lines) of the log file
                    # This is fast and memory-efficient
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                        tmp_path = tmp.name

                    # Use tail to get last 500K lines (roughly 50MB)
                    subprocess.run(
                        ['tail', '-n', '500000', str(self.log_path)],
                        stdout=open(tmp_path, 'w'),
                        check=True
                    )

                    # Replace original file with truncated version
                    import shutil
                    shutil.move(tmp_path, str(self.log_path))

                    new_size_mb = self.log_path.stat().st_size / (1024 * 1024)
                    print(f"   Truncated log to last 500K lines ({new_size_mb:.1f}MB, was {file_size_mb:.1f}MB)")
                    return

                except Exception as tail_error:
                    print(f"   Warning: Tail-based truncation failed: {tail_error}")
                    print(f"   Consider manually deleting or truncating: {self.log_path}")
                    return

            # For smaller files (<100MB), use run-based rotation
            # Read entire log file
            with open(self.log_path) as f:
                content = f.read()

            # Split by run separators (looking for common patterns)
            # Typical separator: "========== Analysis Run: 2025-01-..." or similar
            runs = []
            current_run = []

            for line in content.split("\n"):
                # Detect run separator (adjust pattern based on actual log format)
                if self._is_run_separator(line):
                    if current_run:
                        runs.append("\n".join(current_run))
                    current_run = [line]
                else:
                    current_run.append(line)

            # Don't forget last run
            if current_run:
                runs.append("\n".join(current_run))

            # Keep only last max_runs
            if len(runs) > self.max_runs:
                runs_to_keep = runs[-self.max_runs :]

                # Write back
                with open(self.log_path, "w") as f:
                    f.write("\n".join(runs_to_keep))

                # Log rotation is silent - only shown in verbose mode

        except Exception as e:
            print(f"Warning: Could not rotate log file: {e}")

    def _is_run_separator(self, line: str) -> bool:
        """
        Detect if a line is a run separator.

        Args:
            line: Line to check

        Returns:
            True if line is a run separator
        """
        # Common patterns for run separators
        # Format: "Analysis Run: 2025-10-30 11:10:04 - Interactive Mode"
        separators = [
            "Analysis Run:",
            "========== Analysis Run:",
            "=" * 40 + " NEW RUN ",
            "Analysis started at:",
        ]

        line = line.strip()

        for sep in separators:
            if line.startswith(sep):
                return True

        # Check for timestamp-based separators
        if line.startswith("=") and any(str(year) in line for year in range(2020, 2030)):
            return True

        return False

    def add_run_separator(self, message: str = ""):
        """
        Add a run separator to the log file.

        Args:
            message: Optional message to include in separator
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(self.log_path, "a") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Analysis Run: {timestamp}")
                if message:
                    f.write(f" - {message}")
                f.write("\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            print(f"Warning: Could not write run separator: {e}")


def cleanup_old_files(file_pattern: str, keep_last: int = 100):
    """
    Clean up old files matching a pattern, keeping only the most recent.

    Args:
        file_pattern: Glob pattern for files to clean up
        keep_last: Number of most recent files to keep
    """
    import glob

    files = sorted(glob.glob(file_pattern), key=lambda x: os.path.getmtime(x), reverse=True)

    if len(files) > keep_last:
        files_to_delete = files[keep_last:]

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Removed old file: {file_path}")
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")

        print(f"Cleaned up {len(files_to_delete)} old files")
