/**
 * SQLite Parser for Climb Analyzer
 *
 * Uses sql.js (WebAssembly SQLite) to parse .sqlite databases in the browser.
 * Maps iOS-compatible schema columns to the Climb interface.
 */

import initSqlJs, { Database } from 'sql.js';
import { Climb, ClimbCategory } from '@/types/climb';

// Cache the SQL.js initialization promise
let sqlJsPromise: Promise<any> | null = null;

/**
 * Initialize sql.js with WebAssembly
 * Cached to avoid re-initialization
 */
async function initSqlJsOnce(): Promise<any> {
  if (!sqlJsPromise) {
    sqlJsPromise = initSqlJs({
      // Serve from /public so it works offline and isn't blocked by CDN issues.
      // We copy node_modules/sql.js/dist/sql-wasm.wasm → public/sql-wasm.wasm at build time.
      locateFile: (file: string) => `/${file}`,
    });
  }
  return sqlJsPromise;
}

/**
 * Parse a SQLite database file and extract climbs
 *
 * @param arrayBuffer - The SQLite file contents as ArrayBuffer
 * @returns Array of Climb objects
 */
export async function parseClimbSQLite(arrayBuffer: ArrayBuffer): Promise<Climb[]> {
  const SQL = await initSqlJsOnce();
  const db: Database = new SQL.Database(new Uint8Array(arrayBuffer));

  try {
    // Query all climbs from the database
    const result = db.exec(`
      SELECT
        id,
        streetName,
        city,
        state,
        country,
        distanceFromCenter,
        lat,
        lon,
        cyclingAccess,
        category,
        basicScore,
        fietsScore,
        pdiScore,
        elevationGain,
        height,
        prominence,
        length,
        avgGrade,
        maxGrade,
        highwayType,
        surface,
        tracktype,
        wayId,
        osmLink,
        allWayIds,
        connectedClimbs,
        elevationProfile,
        elevationProfileUnits
      FROM climbs
      ORDER BY pdiScore DESC
    `);

    if (result.length === 0) {
      return [];
    }

    const columns = result[0].columns;
    const values = result[0].values;

    // Map column indices for faster access
    const colIndex: { [key: string]: number } = {};
    columns.forEach((col, index) => {
      colIndex[col] = index;
    });

    // Convert rows to Climb objects
    const climbs: Climb[] = values.map((row) => {
      const getValue = (col: string): any => {
        const idx = colIndex[col];
        return idx !== undefined ? row[idx] : null;
      };

      const getNumber = (col: string): number => {
        const val = getValue(col);
        if (val === null || val === undefined || val === '') return 0;
        const num = parseFloat(val);
        return isNaN(num) ? 0 : num;
      };

      const getString = (col: string): string => {
        const val = getValue(col);
        return val !== null && val !== undefined ? String(val) : '';
      };

      // Map category string to ClimbCategory type
      const categoryStr = getString('category');
      let category: ClimbCategory = 'Uncategorized';
      if (categoryStr === 'HC' || categoryStr === 'hc') {
        category = 'HC';
      } else if (categoryStr === 'Cat 1' || categoryStr === '1') {
        category = 'Cat 1';
      } else if (categoryStr === 'Cat 2' || categoryStr === '2') {
        category = 'Cat 2';
      } else if (categoryStr === 'Cat 3' || categoryStr === '3') {
        category = 'Cat 3';
      } else if (categoryStr === 'Cat 4' || categoryStr === '4') {
        category = 'Cat 4';
      } else if (categoryStr === 'N/A' || categoryStr === '') {
        category = 'Uncategorized';
      }

      return {
        streetName: getString('streetName'),
        city: getString('city'),
        state: getString('state'),
        country: getString('country'),
        distanceFromCenter: getNumber('distanceFromCenter'),
        lat: getNumber('lat'),
        lon: getNumber('lon'),
        category,
        cyclingAccess: getString('cyclingAccess'),
        basicScore: getNumber('basicScore'),
        fietsScore: getNumber('fietsScore'),
        pdiScore: getNumber('pdiScore'),
        elevationGain: getNumber('elevationGain'),
        height: getNumber('height'),
        prominence: getNumber('prominence'),
        length: getNumber('length'),
        avgGrade: getNumber('avgGrade'),
        maxGrade: getNumber('maxGrade'),
        highwayType: getString('highwayType'),
        surface: getString('surface'),
        tracktype: getString('tracktype'),
        wayId: getString('wayId'),
        osmLink: getString('osmLink'),
        connectedClimbs: getString('connectedClimbs'),
        elevationProfile: getString('elevationProfile') || undefined,
        // Store allWayIds in the existing field - will be used by detail drawer
        allWayIds: getString('allWayIds'),
      };
    });

    return climbs;
  } finally {
    db.close();
  }
}

/**
 * Get file statistics from SQLite database
 *
 * @param arrayBuffer - The SQLite file contents as ArrayBuffer
 * @returns File statistics object or null if not available
 */
export async function getFileStats(arrayBuffer: ArrayBuffer): Promise<{
  climbCount: number;
  maxLength: number;
  maxProminence: number;
  maxElevGain: number;
  maxAvgGrade: number;
  maxMaxGrade: number;
  maxHeight: number;
  maxBasicScore: number;
  maxFietsScore: number;
  maxPdiScore: number;
} | null> {
  const SQL = await initSqlJsOnce();
  const db: Database = new SQL.Database(new Uint8Array(arrayBuffer));

  try {
    const result = db.exec(`
      SELECT
        climbCount,
        maxLength,
        maxProminence,
        maxElevGain,
        maxAvgGrade,
        maxMaxGrade,
        maxHeight,
        maxBasicScore,
        maxFietsScore,
        maxPdiScore
      FROM file_stats
      LIMIT 1
    `);

    if (result.length === 0 || result[0].values.length === 0) {
      return null;
    }

    const row = result[0].values[0];
    const columns = result[0].columns;
    const colIndex: { [key: string]: number } = {};
    columns.forEach((col, index) => {
      colIndex[col] = index;
    });

    const getValue = (col: string): number => {
      const idx = colIndex[col];
      const val = idx !== undefined ? row[idx] : null;
      if (val === null || val === undefined) return 0;
      const num = parseFloat(val as string);
      return isNaN(num) ? 0 : num;
    };

    return {
      climbCount: getValue('climbCount'),
      maxLength: getValue('maxLength'),
      maxProminence: getValue('maxProminence'),
      maxElevGain: getValue('maxElevGain'),
      maxAvgGrade: getValue('maxAvgGrade'),
      maxMaxGrade: getValue('maxMaxGrade'),
      maxHeight: getValue('maxHeight'),
      maxBasicScore: getValue('maxBasicScore'),
      maxFietsScore: getValue('maxFietsScore'),
      maxPdiScore: getValue('maxPdiScore'),
    };
  } finally {
    db.close();
  }
}

/**
 * Query climbs within a bounding box using R-tree spatial index
 *
 * @param arrayBuffer - The SQLite file contents as ArrayBuffer
 * @param bounds - Bounding box {minLat, maxLat, minLon, maxLon}
 * @returns Array of Climb objects within the bounds
 */
export async function queryClimbsInBounds(
  arrayBuffer: ArrayBuffer,
  bounds: { minLat: number; maxLat: number; minLon: number; maxLon: number }
): Promise<Climb[]> {
  const SQL = await initSqlJsOnce();
  const db: Database = new SQL.Database(new Uint8Array(arrayBuffer));

  try {
    // Use R-tree spatial index for efficient bounding box query
    const result = db.exec(`
      SELECT c.*
      FROM climbs c
      JOIN climb_rtree_map m ON c.id = m.climb_id
      JOIN climbs_rtree r ON m.rtree_id = r.id
      WHERE r.minLat >= ? AND r.maxLat <= ?
        AND r.minLon >= ? AND r.maxLon <= ?
      ORDER BY c.pdiScore DESC
    `, [bounds.minLat, bounds.maxLat, bounds.minLon, bounds.maxLon]);

    if (result.length === 0) {
      return [];
    }

    // Reuse the parsing logic from parseClimbSQLite
    // For now, return empty - this is an optimization for later
    return [];
  } finally {
    db.close();
  }
}
