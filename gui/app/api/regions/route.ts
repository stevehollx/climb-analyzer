import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const GEO_DEFINITIONS_PATH = path.join(process.cwd(), '..', 'climb_analyzer', 'data', 'geo_definitions.py');

interface Region {
  name: string;
  path: string;
  type: 'continent' | 'country' | 'state' | 'subregion';
  children?: Region[];
}

function parseNestedStructure(content: string): Region[] {
  // Parse the osm_pbf_urls structure recursively
  const regions: Region[] = [];

  // Find osm_pbf_urls block
  const osmUrlsMatch = content.match(/osm_pbf_urls\s*=\s*\{([\s\S]*)\n\}/m);
  if (!osmUrlsMatch) {
    return regions;
  }

  const osmContent = osmUrlsMatch[1];

  // Define continent display names
  const continentNames: Record<string, string> = {
    'africa': 'Africa',
    'asia': 'Asia',
    'europe': 'Europe',
    'north-america': 'North America',
    'south-america': 'South America',
    'central-america': 'Central America',
    'australia-oceania': 'Oceania',
    'antarctica': 'Antarctica',
  };

  // Parse top-level continents
  // Format: "continent": { ... "subregions": { ... } }
  const continentPattern = /"([a-z\-]+)":\s*\{[^}]*?"subregions":\s*\{/g;
  const continentMatches = [...osmContent.matchAll(continentPattern)];

  for (const continentMatch of continentMatches) {
    const continentKey = continentMatch[1];

    // Find the subregions block for this continent
    const continentIndex = continentMatch.index!;
    const subregionsStart = osmContent.indexOf('"subregions":', continentIndex);

    if (subregionsStart === -1) continue;

    // Find matching closing brace for subregions
    let braceCount = 0;
    let subregionsEnd = subregionsStart;
    for (let i = subregionsStart; i < osmContent.length; i++) {
      if (osmContent[i] === '{') braceCount++;
      if (osmContent[i] === '}') {
        braceCount--;
        if (braceCount === 0) {
          subregionsEnd = i;
          break;
        }
      }
    }

    const subregionsContent = osmContent.substring(subregionsStart, subregionsEnd + 1);

    // Extract all region paths - note paths can be either "continent/country" or "country/subregion"
    const regionPattern = /"([a-z\-]+\/[a-z\-]+(?:\/[a-z\-]+)?)":\s*\{/g;
    const regionMatches = [...subregionsContent.matchAll(regionPattern)];

    // Build tree structure
    const continentRegion: Region = {
      name: continentNames[continentKey] || continentKey.charAt(0).toUpperCase() + continentKey.slice(1).replace(/-/g, ' '),
      path: continentKey,
      type: 'continent',
      children: [],
    };

    // Group regions by parent
    const regionMap = new Map<string, Region>();

    // Helper function to format display names with proper capitalization
    const formatDisplayName = (name: string): string => {
      return name.split('-').map(w => {
        // Special case: "us" should be "US" not "Us"
        if (w.toLowerCase() === 'us') {
          return 'US';
        }
        return w.charAt(0).toUpperCase() + w.slice(1);
      }).join(' ');
    };

    // First pass: collect all countries (paths starting with continent/)
    for (const regionMatch of regionMatches) {
      const fullPath = regionMatch[1];
      const parts = fullPath.split('/');

      if (parts.length === 2 && parts[0] === continentKey) {
        // This is a country: "asia/japan" or "asia/china"
        const country = parts[1];
        const displayName = formatDisplayName(country);

        // Check if this country has its own subregions block
        // Look for patterns like "china/beijing" (without continent prefix)
        const hasSubregions = regionMatches.some(m => {
          const mParts = m[1].split('/');
          return mParts.length === 2 && mParts[0] === country;
        });

        const region: Region = {
          name: displayName,
          path: fullPath,
          type: 'country',
          children: hasSubregions ? [] : undefined,
        };

        regionMap.set(country, region); // Store by country name for nested lookups
        continentRegion.children!.push(region);
      }
    }

    // Second pass: add subregions under their parent countries
    for (const regionMatch of regionMatches) {
      const fullPath = regionMatch[1];
      const parts = fullPath.split('/');

      if (parts.length === 2 && parts[0] !== continentKey) {
        // This is a subregion like "china/beijing" or "us/alabama" (not starting with continent)
        const [parentKey, subregion] = parts;
        const parent = regionMap.get(parentKey);

        if (parent && parent.children) {
          const displayName = formatDisplayName(subregion);
          const fullSubregionPath = `${continentKey}/${parentKey}/${subregion}`;

          // Check if this subregion has its own children (e.g., california has norcal/socal)
          const hasSubregions = regionMatches.some(m => {
            const mParts = m[1].split('/');
            return mParts.length === 2 && mParts[0] === subregion;
          });

          const region: Region = {
            name: displayName,
            path: fullSubregionPath,
            type: 'subregion',
            children: hasSubregions ? [] : undefined,
          };

          parent.children.push(region);

          // Store in map for potential grandchildren (e.g., california's norcal/socal)
          regionMap.set(subregion, region);
        }
      }
    }

    // Third pass: add sub-subregions (e.g., california/norcal under us/california)
    for (const regionMatch of regionMatches) {
      const fullPath = regionMatch[1];
      const parts = fullPath.split('/');

      if (parts.length === 2 && parts[0] !== continentKey) {
        const [possibleParent, possibleChild] = parts;
        const parent = regionMap.get(possibleParent);

        // Check if this parent is itself a subregion (not a country)
        if (parent && parent.type === 'subregion' && parent.children) {
          const displayName = formatDisplayName(possibleChild);

          // Construct full path: need to find parent's path and append
          const fullChildPath = `${parent.path}/${possibleChild}`;

          parent.children.push({
            name: displayName,
            path: fullChildPath,
            type: 'subregion',
          });
        }
      }
    }

    // Sort children alphabetically at all levels
    const sortChildren = (region: Region) => {
      if (region.children && region.children.length > 0) {
        region.children.sort((a, b) => a.name.localeCompare(b.name));
        region.children.forEach(child => sortChildren(child));
      }
    };
    sortChildren(continentRegion);

    if (continentRegion.children!.length > 0) {
      regions.push(continentRegion);
    }
  }

  return regions;
}

export async function GET(): Promise<Response> {
  try {
    // Read geo_definitions.py
    const content = await fs.readFile(GEO_DEFINITIONS_PATH, 'utf-8');

    // Parse osm_pbf_urls structure (US states are already included here under north-america/us)
    const regions = parseNestedStructure(content);

    // Sort continents
    regions.sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json({
      regions,
      success: true,
    });
  } catch (error) {
    console.error('Failed to parse regions:', error);
    return NextResponse.json(
      { error: 'Failed to load regions', details: String(error) },
      { status: 500 }
    );
  }
}
