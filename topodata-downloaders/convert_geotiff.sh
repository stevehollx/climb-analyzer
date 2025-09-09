#!/bin/bash

# Convert Geotiff
# Converts hgt zip files to geotiff with gdal_translate
# This is not required but offers 10x read performance for the opentopodata API

#Syntax:
# ./convert_geotiff.sh -v -p '*.hgt.zip' -c 'gdal_translate -co COMPRESS=DEFLATE -co PREDICTOR=2 {hgtzip_filename} {tif_filename}'

# Default values
FOLDER="."
COMMAND=""
RECURSIVE=false
VERBOSE=false
FILE_PATTERN="*"
PARALLEL=false
MAX_JOBS=4

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] -c 'COMMAND'"
    echo ""
    echo "Run a command on every file in a folder"
    echo ""
    echo "Options:"
    echo "  -f, --folder PATH      Folder to process (default: current directory)"
    echo "  -c, --command CMD      Command to run on each file (use {} as placeholder for filename)"
    echo "  -r, --recursive        Process files recursively in subdirectories"
    echo "  -p, --pattern PATTERN  File pattern to match (default: *)"
    echo "  -j, --parallel         Run commands in parallel"
    echo "  -n, --jobs NUM         Number of parallel jobs (default: 4)"
    echo "  -v, --verbose          Verbose output"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -c 'echo \"Processing: {}\"'"
    echo "  $0 -f /path/to/files -c 'gdalinfo {}'"
    echo "  $0 -r -p '*.tif' -c 'gdal_translate {} processed/{}.png'"
    echo "  $0 -j -n 8 -c 'python process_file.py {}'"
    echo "  $0 -p '*.hgt' -c 'gdal_translate -of GTiff {} {}.tif'"
}

# Function to log messages
log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

# Function to log errors
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to log success
success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to log warnings
warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to process a single file
process_file() {
    local file="$1"
    local cmd="$2"
    
    # Get the base filename without extension(s)
    local basename=$(basename "$file")
    local dirname=$(dirname "$file")
    
    # Handle different file extensions and create output filename
    local output_file=""
    if [[ "$basename" == *.hgt.zip ]]; then
        # Remove .hgt.zip and add .tif
        local base_name="${basename%.hgt.zip}"
        output_file="$dirname/${base_name}.tif"
    elif [[ "$basename" == *.hgt ]]; then
        # Remove .hgt and add .tif
        local base_name="${basename%.hgt}"
        output_file="$dirname/${base_name}.tif"
    elif [[ "$basename" == *.zip ]]; then
        # Remove .zip and add .tif
        local base_name="${basename%.zip}"
        output_file="$dirname/${base_name}.tif"
    else
        # Default: remove extension and add .tif
        local base_name="${basename%.*}"
        output_file="$dirname/${base_name}.tif"
    fi
    
    # Replace placeholders in command
    local actual_cmd="$cmd"
    actual_cmd="${actual_cmd//\{hgtzip_filename\}/$file}"
    actual_cmd="${actual_cmd//\{tif_filename\}/$output_file}"
    actual_cmd="${actual_cmd//\{\}/$file}"  # Keep original {} placeholder support
    
    log "Processing: $file"
    log "Output: $output_file"
    log "Command: $actual_cmd"
    
    # Create output directory if it doesn't exist
    local output_dir=$(dirname "$output_file")
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi
    
    if eval "$actual_cmd"; then
        success "Completed: $file -> $output_file"
        return 0
    else
        error "Failed: $file"
        return 1
    fi
}

# Function to process files in parallel
process_parallel() {
    local files=("$@")
    local pids=()
    local active_jobs=0
    local total_files=${#files[@]}
    local processed=0
    local failed=0
    
    echo "Processing $total_files files with up to $MAX_JOBS parallel jobs..."
    
    for file in "${files[@]}"; do
        # Wait if we've reached max jobs
        while [ $active_jobs -ge $MAX_JOBS ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    if [ $? -eq 0 ]; then
                        ((processed++))
                    else
                        ((failed++))
                    fi
                    unset "pids[$i]"
                    ((active_jobs--))
                fi
            done
            sleep 0.1
        done
        
        # Start new job
        process_file "$file" "$COMMAND" &
        pids+=($!)
        ((active_jobs++))
    done
    
    # Wait for remaining jobs
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            if [ $? -eq 0 ]; then
                ((processed++))
            else
                ((failed++))
            fi
        fi
    done
    
    echo ""
    success "Processed: $processed files"
    if [ $failed -gt 0 ]; then
        error "Failed: $failed files"
    fi
}

# Function to process files sequentially
process_sequential() {
    local files=("$@")
    local total_files=${#files[@]}
    local processed=0
    local failed=0
    local current=0
    
    echo "Processing $total_files files sequentially..."
    
    for file in "${files[@]}"; do
        ((current++))
        echo "[$current/$total_files] Processing: $(basename "$file")"
        
        if process_file "$file" "$COMMAND"; then
            ((processed++))
        else
            ((failed++))
        fi
    done
    
    echo ""
    success "Processed: $processed files"
    if [ $failed -gt 0 ]; then
        error "Failed: $failed files"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--folder)
            FOLDER="$2"
            shift 2
            ;;
        -c|--command)
            COMMAND="$2"
            shift 2
            ;;
        -r|--recursive)
            RECURSIVE=true
            shift
            ;;
        -p|--pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        -j|--parallel)
            PARALLEL=true
            shift
            ;;
        -n|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$COMMAND" ]; then
    error "Command is required. Use -c or --command to specify it."
    usage
    exit 1
fi

if [ ! -d "$FOLDER" ]; then
    error "Folder does not exist: $FOLDER"
    exit 1
fi

# Check if {} placeholder is in command
if [[ "$COMMAND" != *"{}"* ]]; then
    warning "Command does not contain {} placeholder. Files will not be passed as arguments."
fi

echo "Batch File Processor"
echo "===================="
echo "Folder: $FOLDER"
echo "Command: $COMMAND"
echo "Pattern: $FILE_PATTERN"
echo "Recursive: $RECURSIVE"
echo "Parallel: $PARALLEL"
if [ "$PARALLEL" = true ]; then
    echo "Max Jobs: $MAX_JOBS"
fi
echo ""

# Find files
files=()
if [ "$RECURSIVE" = true ]; then
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$FOLDER" -name "$FILE_PATTERN" -type f -print0)
else
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$FOLDER" -maxdepth 1 -name "$FILE_PATTERN" -type f -print0)
fi

# Check if any files found
if [ ${#files[@]} -eq 0 ]; then
    warning "No files found matching pattern '$FILE_PATTERN' in $FOLDER"
    exit 0
fi

# Process files
if [ "$PARALLEL" = true ]; then
    process_parallel "${files[@]}"
else
    process_sequential "${files[@]}"
fi

echo "Done!"