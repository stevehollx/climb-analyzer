import { Climb, ClimbCategory } from '@/types/climb';
import * as XLSX from 'xlsx';

/**
 * Parse Excel file (ArrayBuffer) into Climb objects
 */
export function parseClimbExcel(fileBuffer: ArrayBuffer): Climb[] {
  try {
    const workbook = XLSX.read(fileBuffer, { type: 'array' });
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
    const jsonData = XLSX.utils.sheet_to_json(firstSheet, { header: 1 }) as string[][];

    if (jsonData.length < 2) return [];

    const headers = jsonData[0];

    const climbs: Climb[] = [];

    for (let i = 1; i < jsonData.length; i++) {
      const values = jsonData[i];
      const climb: any = {};

      headers.forEach((header, index) => {
        climb[header] = values[index];
      });

      const mappedClimb = mapToClimb(climb);
      climbs.push(mappedClimb);
    }

    return climbs;
  } catch (error) {
    console.error('Error parsing Excel file:', error);
    throw new Error('Failed to parse Excel file');
  }
}

/**
 * Parse CSV string into Climb objects
 */
export function parseClimbCSV(csvContent: string): Climb[] {
  const lines = csvContent.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const climbs: Climb[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length !== headers.length) {
      continue;
    }

    const climb: any = {};
    headers.forEach((header, index) => {
      climb[header] = values[index];
    });

    const mappedClimb = mapToClimb(climb);
    climbs.push(mappedClimb);
  }

  return climbs;
}

/**
 * Parse a single CSV line, handling quoted fields
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}

/**
 * Helper to get value from row with multiple possible column names
 */
function getRowValue(row: any, ...possibleNames: string[]): string {
  for (const name of possibleNames) {
    if (row[name] !== undefined && row[name] !== null && row[name] !== '') {
      return row[name];
    }
  }
  return '';
}

/**
 * Map CSV row object to Climb interface
 */
function mapToClimb(row: any): Climb {
  // Handle different possible column name variations
  const streetName = getRowValue(row, 'Street Name', 'street_name', 'name', 'Name');
  const city = getRowValue(row, 'City', 'city');
  const state = getRowValue(row, 'State', 'state');
  const country = getRowValue(row, 'Country', 'country');
  const lat = parseFloat(getRowValue(row, 'Latitude', 'Lat', 'lat', 'latitude') || '0');
  const lon = parseFloat(getRowValue(row, 'Longitude', 'Lon', 'lon', 'longitude') || '0');
  const category = getRowValue(row, 'Category', 'category', 'cat') || 'Uncategorized';

  // Try to extract numeric values, handling both string and number types
  const parseNumeric = (value: any): number => {
    if (typeof value === 'number') return value;
    if (typeof value === 'string') {
      const parsed = parseFloat(value.replace(/[^0-9.-]/g, ''));
      return isNaN(parsed) ? 0 : parsed;
    }
    return 0;
  };

  const elevGain = getRowValue(row, 'Elev Gain (ft)', 'Elev Gain', 'elev_gain', 'elevation_gain', 'elevation', 'Elevation Gain');
  const lengthVal = getRowValue(row, 'Length (mi)', 'Length', 'length', 'distance', 'Distance');
  const avgGradeVal = getRowValue(row, 'Avg Grade (%)', 'Avg Grade', 'avg_grade', 'average_grade', 'grade', 'Average Grade');
  const maxGradeVal = getRowValue(row, 'Max Grade (%)', 'Max Grade', 'max_grade', 'maximum_grade', 'Maximum Grade');
  const basicScoreVal = getRowValue(row, 'Basic Score', 'basic_score', 'score', 'Score');
  const fietsScoreVal = getRowValue(row, 'FIETS Score', 'fiets_score', 'fiets', 'FIETS');
  const pdiScoreVal = getRowValue(row, 'PDI Score', 'pdi_score', 'pdi', 'PDI');
  const heightVal = getRowValue(row, 'Height (ft)', 'Height', 'height');
  const prominenceVal = getRowValue(row, 'Prominence (ft)', 'Prominence', 'prominence');
  const distanceVal = getRowValue(row, 'From Center (mi)', 'Distance from Center', 'distance_from_center', 'distance');

  return {
    streetName,
    city,
    state,
    country,
    distanceFromCenter: parseNumeric(distanceVal),
    lat,
    lon,
    category: category as ClimbCategory,
    cyclingAccess: getRowValue(row, 'Cycling', 'cycling', 'cycling_access'),
    basicScore: parseNumeric(basicScoreVal),
    fietsScore: parseNumeric(fietsScoreVal),
    pdiScore: parseNumeric(pdiScoreVal),
    elevationGain: parseNumeric(elevGain),
    height: parseNumeric(heightVal),
    prominence: parseNumeric(prominenceVal),
    length: parseNumeric(lengthVal),
    avgGrade: parseNumeric(avgGradeVal),
    maxGrade: parseNumeric(maxGradeVal),
    highwayType: getRowValue(row, 'Highway Type', 'highway_type', 'highway'),
    surface: getRowValue(row, 'Surface', 'surface'),
    tracktype: getRowValue(row, 'Tracktype', 'tracktype', 'track_type'),
    wayId: getRowValue(row, 'Way ID', 'Way ID', 'way_id', 'osm_id', 'id'),
    osmLink: getRowValue(row, 'OSM Link', 'osm_link', 'link'),
    connectedClimbs: getRowValue(row, 'Connected Climbs', 'connected_climbs', 'connected'),
    elevationProfile: getRowValue(row, 'Elevation Profile', 'elevation_profile', 'elev_profile') || undefined,
  };
}

/**
 * Get color for climb category
 */
export function getCategoryColor(category: ClimbCategory | string): string {
  // Normalize category to handle both formats ('1' and 'Cat 1', 'N/A' and 'Uncategorized')
  const categoryMap: { [key: string]: string } = {
    '1': 'Cat 1',
    '2': 'Cat 2',
    '3': 'Cat 3',
    '4': 'Cat 4',
    'N/A': 'Uncategorized'
  };

  const normalizedCategory = categoryMap[category] || category;

  const colors: Record<string, string> = {
    'HC': '#8B0000',      // Dark red
    'Cat 1': '#DC143C',   // Crimson
    'Cat 2': '#FF6347',   // Tomato
    'Cat 3': '#FFA500',   // Orange
    'Cat 4': '#FFD700',   // Gold
    'Uncategorized': '#A9A9A9', // Dark gray
  };

  return colors[normalizedCategory] || colors['Uncategorized'];
}

/**
 * Get opacity for climb category
 */
export function getCategoryOpacity(category: ClimbCategory): number {
  const opacities: Record<ClimbCategory, number> = {
    'HC': 1.0,
    'Cat 1': 0.95,
    'Cat 2': 0.85,
    'Cat 3': 0.75,
    'Cat 4': 0.65,
    'Uncategorized': 0.5,
  };

  return opacities[category] || 0.5;
}

/**
 * Sort climbs by score
 */
export function sortClimbsByScore(
  climbs: Climb[],
  scoreType: 'basic' | 'fiets' | 'pdi'
): Climb[] {
  return [...climbs].sort((a, b) => {
    const scoreA = scoreType === 'basic' ? a.basicScore :
                   scoreType === 'fiets' ? a.fietsScore : a.pdiScore;
    const scoreB = scoreType === 'basic' ? b.basicScore :
                   scoreType === 'fiets' ? b.fietsScore : b.pdiScore;
    return scoreB - scoreA;
  });
}

/**
 * Get top N climbs
 */
export function getTopNClimbs(
  climbs: Climb[],
  n: number,
  scoreType: 'basic' | 'fiets' | 'pdi'
): Climb[] {
  const sorted = sortClimbsByScore(climbs, scoreType);
  return sorted.slice(0, n);
}

/**
 * Get top percentage of climbs
 */
export function getTopPercentClimbs(
  climbs: Climb[],
  percent: number,
  scoreType: 'basic' | 'fiets' | 'pdi'
): Climb[] {
  const count = Math.ceil(climbs.length * (percent / 100));
  return getTopNClimbs(climbs, count, scoreType);
}
