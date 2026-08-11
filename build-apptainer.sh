#!/bin/bash
#
# Build script for CVRmap Apptainer container
#
# Usage:
#   ./build-apptainer.sh [OPTIONS]
#
# Options:
#   --sandbox    Build as a writable sandbox directory (for development/debugging)
#   --fakeroot   Build without root privileges using fakeroot
#   -o, --output PATH   Specify output destination for the container (default: ar ovai.cvrmap.<version>.sif)
#   -h, --help   Display this help message and exit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Extract version from cvrmap/__init__.py (single source of truth)
VERSION=$(grep -oP "__version__\s*=\s*\"\K[^\"]*" "${SCRIPT_DIR}/cvrmap/__init__.py")
DEFAULT_CONTAINER_NAME="arovai.cvrmap.${VERSION}.sif"

# Function to display help
display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build script for CVRmap Apptainer container."
    echo ""
    echo "Options:"
    echo "  --sandbox     Build as a writable sandbox directory (for development/debugging)"
    echo "  --fakeroot    Build without root privileges using fakeroot"
    echo "  -o, --output  PATH   Specify output destination for the container"
    echo "                  (default: ar ovai.cvrmap.<version>.sif)"
    echo "  -h, --help    Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0                          # Build standard SIF container"
    echo "  $0 --sandbox               # Build sandbox directory"
    echo "  $0 --fakeroot              # Build with fakeroot"
    echo "  $0 --sandbox --fakeroot    # Build sandbox with fakeroot"
    echo "  $0 --output /custom/path/container.sif   # Custom output path"
}

# Parse arguments
SANDBOX=false
FAKEROOT=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sandbox)
            SANDBOX=true
            shift
            ;;
        --fakeroot)
            FAKEROOT="--fakeroot"
            shift
            ;;
        -o|--output)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --output requires a path argument"
                exit 1
            fi
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Building CVRmap Apptainer Container"
echo "Version: ${VERSION}"
echo "========================================"

cd "${SCRIPT_DIR}"

# Determine the output target
if [ -n "$OUTPUT_PATH" ]; then
    TARGET_PATH="$OUTPUT_PATH"
else
    if [ "$SANDBOX" = true ]; then
        TARGET_PATH="cvrmap_${VERSION}_sandbox"
        DEFAULT_NAME_MSG="(sandbox directory)"
    else
        TARGET_PATH="${DEFAULT_CONTAINER_NAME}"
        DEFAULT_NAME_MSG="(default filename)"
    fi
fi

if [ "$SANDBOX" = true ]; then
    echo "Building as sandbox..."
    echo "Target: ${TARGET_PATH} ${DEFAULT_NAME_MSG:-}"
    apptainer build ${FAKEROOT} --sandbox "${TARGET_PATH}" Apptainer
    echo ""
    echo "Sandbox created: ${TARGET_PATH}/"
    echo "To enter the sandbox: apptainer shell --writable ${TARGET_PATH}/"
else
    echo "Building SIF container..."
    echo "Output: ${TARGET_PATH}"
    apptainer build ${FAKEROOT} "${TARGET_PATH}" Apptainer
    echo ""
    echo "Container built: ${TARGET_PATH}"
fi

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "To run the container:"
echo "  apptainer run ${TARGET_PATH} --help"
echo ""
echo "Example usage:"
echo "  apptainer run ${TARGET_PATH} /data/bids /data/derivatives participant \\"
echo "      --derivatives fmriprep=/data/derivatives/fmriprep \\"
echo "      --task gas --participant_label 001"
