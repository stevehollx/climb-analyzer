#!/bin/bash
#
# Osmium-Tool Installation Script
#
# Installs osmium-tool for merging OSM .pbf files across multiple regions.
# Required when your analysis area spans multiple states/countries.
#

set -e  # Exit on error

echo "======================================================================="
echo "  OSMIUM-TOOL INSTALLATION"
echo "======================================================================="
echo ""
echo "Osmium-tool is required for merging OSM files when your search area"
echo "spans multiple regions (e.g., cross-border climbs)."
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        echo "Detected: Debian/Ubuntu Linux"
        echo "Installing osmium-tool via apt-get..."
        echo ""
        sudo apt-get update
        sudo apt-get install -y osmium-tool
    elif command -v dnf &> /dev/null; then
        echo "Detected: Fedora/RHEL Linux"
        echo "Installing osmium-tool via dnf..."
        echo ""
        sudo dnf install -y osmium-tool
    elif command -v pacman &> /dev/null; then
        echo "Detected: Arch Linux"
        echo "Installing osmium-tool via pacman..."
        echo ""
        sudo pacman -S --noconfirm osmium-tool
    elif command -v zypper &> /dev/null; then
        echo "Detected: openSUSE Linux"
        echo "Installing osmium-tool via zypper..."
        echo ""
        sudo zypper install -y osmium-tool
    else
        echo "❌ Could not detect package manager"
        echo ""
        echo "Please install osmium-tool manually:"
        echo "  • Visit: https://osmcode.org/osmium-tool/"
        echo "  • Or build from source: https://github.com/osmcode/osmium-tool"
        exit 1
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "Detected: macOS with Homebrew"
        echo "Installing osmium-tool via brew..."
        echo ""
        brew install osmium-tool
    else
        echo "❌ Homebrew not found"
        echo ""
        echo "Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo ""
        echo "Then run this script again, or install manually:"
        echo "  brew install osmium-tool"
        exit 1
    fi

else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo ""
    echo "Please install osmium-tool manually:"
    echo "  • Visit: https://osmcode.org/osmium-tool/"
    echo "  • Or build from source: https://github.com/osmcode/osmium-tool"
    exit 1
fi

echo ""
echo "======================================================================="
echo "  ✓ INSTALLATION COMPLETE"
echo "======================================================================="
echo ""

# Verify installation
if command -v osmium &> /dev/null; then
    OSMIUM_VERSION=$(osmium --version | head -n1)
    echo "Osmium-tool successfully installed!"
    echo "Version: $OSMIUM_VERSION"
    echo ""
    echo "You can now run cross-border analyses. The system will automatically:"
    echo "  1. Detect when your search spans multiple regions"
    echo "  2. Download all required OSM files"
    echo "  3. Merge them using osmium-tool"
    echo "  4. Build a unified spatial index"
    echo "  5. Analyze climbs across all regions"
    echo ""
    echo "Re-run your analysis - osmium will be detected automatically."
else
    echo "⚠️  Installation completed but osmium not found in PATH"
    echo "You may need to restart your terminal or add osmium to your PATH."
fi

echo ""
