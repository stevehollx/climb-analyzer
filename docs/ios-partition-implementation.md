# iOS App Partition Support Implementation Guide

This document provides implementation guidance for adding geographic partition support to the iOS app "Climb Visualizer: Pocket". The app needs to handle partitioned SQLite databases for large regions that exceed iOS Safari's WASM memory limits.

## Background

Large regions (California, France, etc.) are now split into multiple SQLite partition files:
- Each partition is under 1.5GB (safe for iOS WASM)
- Partitions have geographic bounds (e.g., NorCal vs SoCal)
- The `index.json` file describes available partitions per region
- Each partition file has a `.metadata.json` companion with bounds, climb count, and SHA256 checksum

## Data Format

### index.json Structure for Partitioned Regions

```json
{
  "regions": {
    "north-america/united-states-of-america/california": {
      "region_name": "California",
      "version": "2.4.0",
      "release_tag": "california-2.4.0",
      "is_partitioned": true,
      "partition_type": "geofabrik",
      "partitions": [
        {
          "partition_id": "norcal",
          "display_name": "Northern California",
          "database_file": "California_climbs_v2.4.0_norcal.sqlite",
          "database_size": 524288000,
          "database_url": "https://github.com/.../California_climbs_v2.4.0_norcal.sqlite",
          "database_sha256": "abc123...",
          "bounds": {
            "minLat": 36.5,
            "minLon": -124.5,
            "maxLat": 42.0,
            "maxLon": -119.0
          },
          "climb_count": 45000,
          "size_mb": 500.0
        },
        {
          "partition_id": "socal",
          "display_name": "Southern California",
          "database_file": "California_climbs_v2.4.0_socal.sqlite",
          "database_size": 419430400,
          "database_url": "https://github.com/.../California_climbs_v2.4.0_socal.sqlite",
          "database_sha256": "def456...",
          "bounds": {
            "minLat": 32.5,
            "minLon": -121.0,
            "maxLat": 36.5,
            "maxLon": -114.0
          },
          "climb_count": 38000,
          "size_mb": 400.0
        }
      ],
      "total_database_size": 943718400
    }
  }
}
```

---

## Phase 1: Data Models

### Task 1.1: Partition and Region Models

Create or update your models to handle partitioned regions.

```swift
// Models/Partition.swift

import Foundation

struct PartitionBounds: Codable, Equatable {
    let minLat: Double
    let minLon: Double
    let maxLat: Double
    let maxLon: Double

    /// Check if a coordinate is within these bounds
    func contains(latitude: Double, longitude: Double) -> Bool {
        return latitude >= minLat && latitude <= maxLat &&
               longitude >= minLon && longitude <= maxLon
    }

    /// Check if these bounds intersect with another bounds
    func intersects(_ other: PartitionBounds) -> Bool {
        return !(maxLat < other.minLat ||
                 minLat > other.maxLat ||
                 maxLon < other.minLon ||
                 minLon > other.maxLon)
    }
}

struct Partition: Codable, Identifiable {
    let partitionId: String
    let displayName: String
    let databaseFile: String
    let databaseSize: Int
    let databaseUrl: String
    let databaseSha256: String?
    let bounds: PartitionBounds?
    let climbCount: Int?
    let sizeMb: Double?

    var id: String { partitionId }

    var formattedSize: String {
        if let mb = sizeMb {
            return String(format: "%.0f MB", mb)
        }
        return ByteCountFormatter.string(fromByteCount: Int64(databaseSize), countStyle: .file)
    }

    var formattedClimbCount: String {
        guard let count = climbCount else { return "" }
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: count)) ?? "\(count)"
    }

    enum CodingKeys: String, CodingKey {
        case partitionId = "partition_id"
        case displayName = "display_name"
        case databaseFile = "database_file"
        case databaseSize = "database_size"
        case databaseUrl = "database_url"
        case databaseSha256 = "database_sha256"
        case bounds
        case climbCount = "climb_count"
        case sizeMb = "size_mb"
    }
}
```

### Task 1.2: Update RegionMetadata Model

```swift
// Models/RegionMetadata.swift

struct RegionMetadata: Codable, Identifiable {
    let regionName: String
    let version: String
    let releaseTag: String
    let climbCount: Int?

    // Single database (non-partitioned)
    let databaseFile: String?
    let databaseSize: Int?
    let databaseUrl: String?

    // Partitioned database
    let isPartitioned: Bool
    let partitionType: String?  // "geofabrik" or "quadtree"
    let partitions: [Partition]?
    let totalDatabaseSize: Int?

    var id: String { releaseTag }

    /// Total size for display (works for both partitioned and non-partitioned)
    var displaySize: String {
        let size = totalDatabaseSize ?? databaseSize ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
    }

    /// Number of partitions (0 if not partitioned)
    var partitionCount: Int {
        return partitions?.count ?? 0
    }

    enum CodingKeys: String, CodingKey {
        case regionName = "region_name"
        case version
        case releaseTag = "release_tag"
        case climbCount = "climb_count"
        case databaseFile = "database_file"
        case databaseSize = "database_size"
        case databaseUrl = "database_url"
        case isPartitioned = "is_partitioned"
        case partitionType = "partition_type"
        case partitions
        case totalDatabaseSize = "total_database_size"
    }
}
```

### Task 1.3: Downloaded Partition Tracking

```swift
// Models/DownloadedPartition.swift

import Foundation

struct DownloadedPartition: Codable, Identifiable {
    let regionKey: String        // e.g., "north-america/united-states-of-america/california"
    let partitionId: String      // e.g., "norcal"
    let localPath: String        // Path relative to documents directory
    let downloadedAt: Date
    let sha256Verified: Bool
    let climbCount: Int?
    let bounds: PartitionBounds?
    let version: String

    var id: String { "\(regionKey)_\(partitionId)" }

    /// Get the full URL to the local file
    func localURL() -> URL? {
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        return documentsDir?.appendingPathComponent(localPath)
    }
}

// MARK: - Persistent Storage

class DownloadedPartitionsStore: ObservableObject {
    @Published private(set) var partitions: [String: [DownloadedPartition]] = [:]  // regionKey -> partitions

    private let userDefaultsKey = "downloadedPartitions"

    init() {
        loadFromStorage()
    }

    // MARK: - Query Methods

    func isPartitionDownloaded(regionKey: String, partitionId: String) -> Bool {
        return partitions[regionKey]?.contains { $0.partitionId == partitionId } ?? false
    }

    func getDownloadedPartition(regionKey: String, partitionId: String) -> DownloadedPartition? {
        return partitions[regionKey]?.first { $0.partitionId == partitionId }
    }

    func getDownloadedPartitions(for regionKey: String) -> [DownloadedPartition] {
        return partitions[regionKey] ?? []
    }

    func getLocalPath(regionKey: String, partitionId: String) -> URL? {
        return getDownloadedPartition(regionKey: regionKey, partitionId: partitionId)?.localURL()
    }

    // MARK: - Mutation Methods

    func addPartition(_ partition: DownloadedPartition) {
        var regionPartitions = partitions[partition.regionKey] ?? []

        // Remove existing entry if present (update)
        regionPartitions.removeAll { $0.partitionId == partition.partitionId }
        regionPartitions.append(partition)

        partitions[partition.regionKey] = regionPartitions
        saveToStorage()
    }

    func removePartition(regionKey: String, partitionId: String) {
        partitions[regionKey]?.removeAll { $0.partitionId == partitionId }

        // Also delete the file
        if let url = getLocalPath(regionKey: regionKey, partitionId: partitionId) {
            try? FileManager.default.removeItem(at: url)
        }

        saveToStorage()
    }

    func removeAllPartitions(for regionKey: String) {
        // Delete all files first
        for partition in getDownloadedPartitions(for: regionKey) {
            if let url = partition.localURL() {
                try? FileManager.default.removeItem(at: url)
            }
        }

        partitions[regionKey] = nil
        saveToStorage()
    }

    // MARK: - Persistence

    private func loadFromStorage() {
        guard let data = UserDefaults.standard.data(forKey: userDefaultsKey),
              let decoded = try? JSONDecoder().decode([String: [DownloadedPartition]].self, from: data) else {
            return
        }
        partitions = decoded
    }

    private func saveToStorage() {
        guard let data = try? JSONEncoder().encode(partitions) else { return }
        UserDefaults.standard.set(data, forKey: userDefaultsKey)
    }
}
```

---

## Phase 2: Download Manager

### Task 2.1: Partition Download Manager with SHA256 Verification

```swift
// Services/PartitionDownloadManager.swift

import Foundation
import CryptoKit

enum DownloadError: LocalizedError {
    case invalidURL
    case downloadFailed(Error)
    case checksumMismatch(expected: String, actual: String)
    case fileMoveFailed(Error)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid download URL"
        case .downloadFailed(let error):
            return "Download failed: \(error.localizedDescription)"
        case .checksumMismatch(let expected, let actual):
            return "Checksum mismatch: expected \(expected.prefix(8))..., got \(actual.prefix(8))..."
        case .fileMoveFailed(let error):
            return "Failed to save file: \(error.localizedDescription)"
        }
    }
}

enum DownloadState: Equatable {
    case idle
    case downloading(progress: Double)
    case verifying
    case completed
    case failed(String)

    static func == (lhs: DownloadState, rhs: DownloadState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.verifying, .verifying), (.completed, .completed):
            return true
        case (.downloading(let p1), .downloading(let p2)):
            return p1 == p2
        case (.failed(let e1), .failed(let e2)):
            return e1 == e2
        default:
            return false
        }
    }
}

@MainActor
class PartitionDownloadManager: ObservableObject {
    @Published private(set) var downloads: [String: DownloadState] = [:]  // partitionId -> state

    private var downloadTasks: [String: URLSessionDownloadTask] = [:]
    private var observations: [String: NSKeyValueObservation] = [:]

    // MARK: - Public API

    func isDownloading(_ partitionId: String) -> Bool {
        if case .downloading = downloads[partitionId] {
            return true
        }
        return false
    }

    func progress(for partitionId: String) -> Double {
        if case .downloading(let progress) = downloads[partitionId] {
            return progress
        }
        return 0
    }

    func state(for partitionId: String) -> DownloadState {
        return downloads[partitionId] ?? .idle
    }

    func cancelDownload(_ partitionId: String) {
        downloadTasks[partitionId]?.cancel()
        downloadTasks[partitionId] = nil
        observations[partitionId]?.invalidate()
        observations[partitionId] = nil
        downloads[partitionId] = .idle
    }

    // MARK: - Download

    func downloadPartition(
        _ partition: Partition,
        for regionKey: String,
        store: DownloadedPartitionsStore
    ) async throws -> URL {
        guard let url = URL(string: partition.databaseUrl) else {
            throw DownloadError.invalidURL
        }

        let partitionId = partition.partitionId
        downloads[partitionId] = .downloading(progress: 0)

        do {
            // Download to temp location
            let tempURL = try await downloadFile(from: url, partitionId: partitionId)

            // Verify SHA256 if available
            if let expectedHash = partition.databaseSha256 {
                downloads[partitionId] = .verifying
                let actualHash = try computeSHA256(of: tempURL)

                guard actualHash.lowercased() == expectedHash.lowercased() else {
                    try? FileManager.default.removeItem(at: tempURL)
                    throw DownloadError.checksumMismatch(expected: expectedHash, actual: actualHash)
                }
            }

            // Move to permanent location
            let finalPath = try moveToDocuments(
                tempURL: tempURL,
                regionKey: regionKey,
                filename: partition.databaseFile
            )

            // Record in store
            let downloaded = DownloadedPartition(
                regionKey: regionKey,
                partitionId: partition.partitionId,
                localPath: "partitions/\(regionKey)/\(partition.databaseFile)",
                downloadedAt: Date(),
                sha256Verified: partition.databaseSha256 != nil,
                climbCount: partition.climbCount,
                bounds: partition.bounds,
                version: partition.databaseFile  // Extract version from filename if needed
            )
            store.addPartition(downloaded)

            downloads[partitionId] = .completed
            return finalPath

        } catch {
            downloads[partitionId] = .failed(error.localizedDescription)
            throw error
        }
    }

    // MARK: - Private Helpers

    private func downloadFile(from url: URL, partitionId: String) async throws -> URL {
        let (tempURL, response) = try await URLSession.shared.download(from: url) { [weak self] totalBytesWritten, totalBytesExpectedToWrite in
            guard totalBytesExpectedToWrite > 0 else { return }
            let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            Task { @MainActor [weak self] in
                self?.downloads[partitionId] = .downloading(progress: progress)
            }
        }

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw DownloadError.downloadFailed(NSError(domain: "HTTP", code: (response as? HTTPURLResponse)?.statusCode ?? 0))
        }

        return tempURL
    }

    private func computeSHA256(of url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }

    private func moveToDocuments(tempURL: URL, regionKey: String, filename: String) throws -> URL {
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let partitionsDir = documentsDir.appendingPathComponent("partitions/\(regionKey)")

        try FileManager.default.createDirectory(at: partitionsDir, withIntermediateDirectories: true)

        let finalPath = partitionsDir.appendingPathComponent(filename)

        // Remove existing file if present
        try? FileManager.default.removeItem(at: finalPath)

        do {
            try FileManager.default.moveItem(at: tempURL, to: finalPath)
        } catch {
            throw DownloadError.fileMoveFailed(error)
        }

        return finalPath
    }
}

// MARK: - URLSession Extension for Progress

extension URLSession {
    func download(from url: URL, progressHandler: @escaping (Int64, Int64) -> Void) async throws -> (URL, URLResponse) {
        try await withCheckedThrowingContinuation { continuation in
            let task = self.downloadTask(with: url) { url, response, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else if let url = url, let response = response {
                    continuation.resume(returning: (url, response))
                } else {
                    continuation.resume(throwing: URLError(.unknown))
                }
            }

            // Observe progress
            let observation = task.progress.observe(\.fractionCompleted) { progress, _ in
                progressHandler(task.countOfBytesReceived, task.countOfBytesExpectedToReceive)
            }

            // Store observation to prevent deallocation
            objc_setAssociatedObject(task, "progressObservation", observation, .OBJC_ASSOCIATION_RETAIN)

            task.resume()
        }
    }
}
```

---

## Phase 3: Database Manager

### Task 3.1: Multi-Partition Database Manager

```swift
// Services/PartitionedDatabaseManager.swift

import Foundation
import SQLite3
import MapKit

class PartitionedDatabaseManager {
    private var connections: [String: OpaquePointer] = [:]  // partitionId -> db connection
    private let queue = DispatchQueue(label: "com.climbvisualizer.db.partition", attributes: .concurrent)

    // MARK: - Connection Management

    /// Load a partition database
    func loadPartition(path: URL, partitionId: String) throws {
        var db: OpaquePointer?
        let result = sqlite3_open_v2(
            path.path,
            &db,
            SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX,
            nil
        )

        guard result == SQLITE_OK, let db = db else {
            throw DatabaseError.openFailed(result)
        }

        queue.async(flags: .barrier) { [weak self] in
            self?.connections[partitionId] = db
        }
    }

    /// Unload a partition to free memory
    func unloadPartition(_ partitionId: String) {
        queue.async(flags: .barrier) { [weak self] in
            if let db = self?.connections[partitionId] {
                sqlite3_close(db)
                self?.connections[partitionId] = nil
            }
        }
    }

    /// Unload all partitions
    func unloadAll() {
        queue.async(flags: .barrier) { [weak self] in
            for (_, db) in self?.connections ?? [:] {
                sqlite3_close(db)
            }
            self?.connections.removeAll()
        }
    }

    /// Get list of loaded partition IDs
    var loadedPartitionIds: [String] {
        queue.sync {
            return Array(connections.keys)
        }
    }

    // MARK: - Queries

    /// Query climbs across all loaded partitions within a map region
    func queryClimbs(
        in region: MKCoordinateRegion,
        limit: Int = 500
    ) -> [Climb] {
        var allClimbs: [Climb] = []

        queue.sync {
            for (partitionId, db) in connections {
                let climbs = queryClimbsInPartition(
                    db: db,
                    region: region,
                    limit: limit
                )
                allClimbs.append(contentsOf: climbs)
            }
        }

        // Sort by PDI score and limit total results
        return allClimbs
            .sorted { $0.pdiScore > $1.pdiScore }
            .prefix(limit)
            .map { $0 }
    }

    /// Query using R-tree spatial index (if available) or bounds filter
    private func queryClimbsInPartition(
        db: OpaquePointer,
        region: MKCoordinateRegion,
        limit: Int
    ) -> [Climb] {
        let minLat = region.center.latitude - region.span.latitudeDelta / 2
        let maxLat = region.center.latitude + region.span.latitudeDelta / 2
        let minLon = region.center.longitude - region.span.longitudeDelta / 2
        let maxLon = region.center.longitude + region.span.longitudeDelta / 2

        // Try R-tree query first, fall back to regular bounds query
        let sql: String
        if hasRtreeIndex(db: db) {
            sql = """
                SELECT c.* FROM climbs c
                INNER JOIN climb_rtree_map m ON c.id = m.climb_id
                INNER JOIN climbs_rtree r ON m.rtree_id = r.id
                WHERE r.minLat <= ? AND r.maxLat >= ?
                  AND r.minLon <= ? AND r.maxLon >= ?
                ORDER BY c.pdiScore DESC
                LIMIT ?
            """
        } else {
            sql = """
                SELECT * FROM climbs
                WHERE startLat >= ? AND startLat <= ?
                  AND startLon >= ? AND startLon <= ?
                ORDER BY pdiScore DESC
                LIMIT ?
            """
        }

        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            return []
        }
        defer { sqlite3_finalize(statement) }

        // Bind parameters
        if hasRtreeIndex(db: db) {
            sqlite3_bind_double(statement, 1, maxLat)  // r.minLat <= maxLat
            sqlite3_bind_double(statement, 2, minLat)  // r.maxLat >= minLat
            sqlite3_bind_double(statement, 3, maxLon)  // r.minLon <= maxLon
            sqlite3_bind_double(statement, 4, minLon)  // r.maxLon >= minLon
        } else {
            sqlite3_bind_double(statement, 1, minLat)
            sqlite3_bind_double(statement, 2, maxLat)
            sqlite3_bind_double(statement, 3, minLon)
            sqlite3_bind_double(statement, 4, maxLon)
        }
        sqlite3_bind_int(statement, 5, Int32(limit))

        var climbs: [Climb] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            if let climb = parseClimbRow(statement: statement) {
                climbs.append(climb)
            }
        }

        return climbs
    }

    private func hasRtreeIndex(db: OpaquePointer) -> Bool {
        let sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='climbs_rtree'"
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }

        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            return false
        }
        return sqlite3_step(statement) == SQLITE_ROW
    }

    private func parseClimbRow(statement: OpaquePointer?) -> Climb? {
        guard let statement = statement else { return nil }

        // Parse columns - adjust indices based on your actual schema
        // This is a simplified example
        let id = Int(sqlite3_column_int64(statement, 0))
        let name = String(cString: sqlite3_column_text(statement, 1))
        let startLat = sqlite3_column_double(statement, 2)
        let startLon = sqlite3_column_double(statement, 3)
        let pdiScore = sqlite3_column_double(statement, 4)
        // ... parse other columns

        return Climb(
            id: id,
            name: name,
            startLat: startLat,
            startLon: startLon,
            pdiScore: pdiScore
            // ... other properties
        )
    }
}

enum DatabaseError: LocalizedError {
    case openFailed(Int32)
    case queryFailed(String)

    var errorDescription: String? {
        switch self {
        case .openFailed(let code):
            return "Failed to open database (code: \(code))"
        case .queryFailed(let message):
            return "Query failed: \(message)"
        }
    }
}
```

### Task 3.2: Lazy Partition Loader

```swift
// Services/LazyPartitionLoader.swift

import Foundation
import MapKit

/// Automatically loads/unloads partitions based on map viewport
class LazyPartitionLoader: ObservableObject {
    @Published private(set) var loadedPartitions: Set<String> = []
    @Published private(set) var isLoading = false

    private let dbManager: PartitionedDatabaseManager
    private let downloadStore: DownloadedPartitionsStore

    init(dbManager: PartitionedDatabaseManager, downloadStore: DownloadedPartitionsStore) {
        self.dbManager = dbManager
        self.downloadStore = downloadStore
    }

    /// Load partitions needed for current viewport
    func loadPartitionsForViewport(
        region: RegionMetadata,
        viewport: MKCoordinateRegion
    ) async {
        guard let partitions = region.partitions else { return }

        await MainActor.run {
            isLoading = true
        }

        defer {
            Task { @MainActor in
                isLoading = false
            }
        }

        // Find partitions that intersect viewport
        let viewportBounds = PartitionBounds(
            minLat: viewport.center.latitude - viewport.span.latitudeDelta / 2,
            minLon: viewport.center.longitude - viewport.span.longitudeDelta / 2,
            maxLat: viewport.center.latitude + viewport.span.latitudeDelta / 2,
            maxLon: viewport.center.longitude + viewport.span.longitudeDelta / 2
        )

        let neededPartitions = partitions.filter { partition in
            guard let bounds = partition.bounds else { return true }  // Load if no bounds info
            return bounds.intersects(viewportBounds)
        }

        let neededIds = Set(neededPartitions.map { $0.partitionId })

        // Unload partitions no longer needed (optional - for memory management)
        for partitionId in loadedPartitions {
            if !neededIds.contains(partitionId) {
                dbManager.unloadPartition(partitionId)
                await MainActor.run {
                    loadedPartitions.remove(partitionId)
                }
            }
        }

        // Load needed partitions that are downloaded but not loaded
        for partition in neededPartitions {
            if !loadedPartitions.contains(partition.partitionId) {
                if let localPath = downloadStore.getLocalPath(
                    regionKey: region.releaseTag,
                    partitionId: partition.partitionId
                ) {
                    do {
                        try dbManager.loadPartition(
                            path: localPath,
                            partitionId: partition.partitionId
                        )
                        await MainActor.run {
                            loadedPartitions.insert(partition.partitionId)
                        }
                    } catch {
                        print("Failed to load partition \(partition.partitionId): \(error)")
                    }
                }
            }
        }
    }

    /// Get partition IDs that cover a viewport but aren't downloaded
    func getMissingPartitions(
        region: RegionMetadata,
        viewport: MKCoordinateRegion
    ) -> [Partition] {
        guard let partitions = region.partitions else { return [] }

        let viewportBounds = PartitionBounds(
            minLat: viewport.center.latitude - viewport.span.latitudeDelta / 2,
            minLon: viewport.center.longitude - viewport.span.longitudeDelta / 2,
            maxLat: viewport.center.latitude + viewport.span.latitudeDelta / 2,
            maxLon: viewport.center.longitude + viewport.span.longitudeDelta / 2
        )

        return partitions.filter { partition in
            guard let bounds = partition.bounds else { return true }
            let intersects = bounds.intersects(viewportBounds)
            let downloaded = downloadStore.isPartitionDownloaded(
                regionKey: region.releaseTag,
                partitionId: partition.partitionId
            )
            return intersects && !downloaded
        }
    }

    /// Unload all partitions
    func unloadAll() {
        dbManager.unloadAll()
        loadedPartitions.removeAll()
    }
}
```

---

## Phase 4: UI Components

### Task 4.1: Partition Picker View

```swift
// Views/PartitionPickerView.swift

import SwiftUI

struct PartitionPickerView: View {
    let region: RegionMetadata
    @ObservedObject var downloadStore: DownloadedPartitionsStore
    @ObservedObject var downloadManager: PartitionDownloadManager

    @State private var downloadError: String?
    @State private var showingError = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text(region.regionName)
                    .font(.title2)
                    .bold()

                Text("This region is split into \(region.partitionCount) geographic areas. Download the areas you need.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                HStack {
                    Text("Total size: \(region.displaySize)")
                    Text("•")
                    Text("\(downloadedCount)/\(region.partitionCount) downloaded")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }

            Divider()

            // Partition list
            if let partitions = region.partitions {
                ForEach(partitions) { partition in
                    PartitionRow(
                        partition: partition,
                        isDownloaded: downloadStore.isPartitionDownloaded(
                            regionKey: region.releaseTag,
                            partitionId: partition.partitionId
                        ),
                        downloadState: downloadManager.state(for: partition.partitionId),
                        onDownload: { downloadPartition(partition) },
                        onDelete: { deletePartition(partition) }
                    )
                }
            }

            Divider()

            // Download All button
            if downloadedCount < region.partitionCount {
                Button(action: downloadAllPartitions) {
                    HStack {
                        Image(systemName: "arrow.down.circle")
                        Text("Download All Partitions")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(isAnyDownloading)

                Text("Downloads all partitions sequentially.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
            } else {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("All partitions downloaded")
                        .fontWeight(.medium)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.green.opacity(0.1))
                .cornerRadius(8)
            }
        }
        .padding()
        .alert("Download Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(downloadError ?? "Unknown error")
        }
    }

    // MARK: - Computed Properties

    private var downloadedCount: Int {
        region.partitions?.filter { partition in
            downloadStore.isPartitionDownloaded(
                regionKey: region.releaseTag,
                partitionId: partition.partitionId
            )
        }.count ?? 0
    }

    private var isAnyDownloading: Bool {
        region.partitions?.contains { partition in
            downloadManager.isDownloading(partition.partitionId)
        } ?? false
    }

    // MARK: - Actions

    private func downloadPartition(_ partition: Partition) {
        Task {
            do {
                _ = try await downloadManager.downloadPartition(
                    partition,
                    for: region.releaseTag,
                    store: downloadStore
                )
            } catch {
                downloadError = error.localizedDescription
                showingError = true
            }
        }
    }

    private func deletePartition(_ partition: Partition) {
        downloadStore.removePartition(
            regionKey: region.releaseTag,
            partitionId: partition.partitionId
        )
    }

    private func downloadAllPartitions() {
        guard let partitions = region.partitions else { return }

        Task {
            for partition in partitions {
                if !downloadStore.isPartitionDownloaded(
                    regionKey: region.releaseTag,
                    partitionId: partition.partitionId
                ) {
                    do {
                        _ = try await downloadManager.downloadPartition(
                            partition,
                            for: region.releaseTag,
                            store: downloadStore
                        )
                    } catch {
                        // Continue with other partitions even if one fails
                        print("Failed to download \(partition.partitionId): \(error)")
                    }
                }
            }
        }
    }
}

// MARK: - Partition Row

struct PartitionRow: View {
    let partition: Partition
    let isDownloaded: Bool
    let downloadState: DownloadState
    let onDownload: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(partition.displayName)
                    .font(.headline)

                HStack(spacing: 8) {
                    if let count = partition.climbCount {
                        Text("\(partition.formattedClimbCount) climbs")
                    }
                    Text(partition.formattedSize)
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }

            Spacer()

            // Status/Action
            Group {
                switch downloadState {
                case .idle:
                    if isDownloaded {
                        Menu {
                            Button(role: .destructive, action: onDelete) {
                                Label("Delete", systemImage: "trash")
                            }
                        } label: {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.title2)
                        }
                    } else {
                        Button(action: onDownload) {
                            Image(systemName: "arrow.down.circle")
                                .font(.title2)
                        }
                    }

                case .downloading(let progress):
                    VStack(spacing: 2) {
                        ProgressView(value: progress)
                            .frame(width: 60)
                        Text("\(Int(progress * 100))%")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }

                case .verifying:
                    HStack(spacing: 4) {
                        ProgressView()
                            .scaleEffect(0.7)
                        Text("Verifying")
                            .font(.caption)
                    }

                case .completed:
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.title2)

                case .failed(let error):
                    Button(action: onDownload) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                            .font(.title2)
                    }
                    .help(error)
                }
            }
        }
        .padding(.vertical, 8)
    }
}
```

---

## Phase 5: Integration

### Task 5.1: Update Region Selection to Show Partitions

When a user selects a partitioned region, show the partition picker instead of immediately downloading.

```swift
// In your region selection view

struct RegionDetailView: View {
    let region: RegionMetadata
    @EnvironmentObject var downloadStore: DownloadedPartitionsStore
    @EnvironmentObject var downloadManager: PartitionDownloadManager

    var body: some View {
        ScrollView {
            if region.isPartitioned {
                // Show partition picker for partitioned regions
                PartitionPickerView(
                    region: region,
                    downloadStore: downloadStore,
                    downloadManager: downloadManager
                )
            } else {
                // Show single download button for non-partitioned regions
                SingleRegionDownloadView(region: region)
            }
        }
        .navigationTitle(region.regionName)
    }
}
```

### Task 5.2: Update Map View to Use Lazy Loading

```swift
// In your map view

struct ClimbMapView: View {
    let region: RegionMetadata
    @StateObject private var lazyLoader: LazyPartitionLoader
    @EnvironmentObject var downloadStore: DownloadedPartitionsStore
    @EnvironmentObject var dbManager: PartitionedDatabaseManager

    @State private var mapRegion: MKCoordinateRegion
    @State private var climbs: [Climb] = []
    @State private var missingPartitions: [Partition] = []

    var body: some View {
        ZStack {
            Map(coordinateRegion: $mapRegion, annotationItems: climbs) { climb in
                // Your climb annotations
            }
            .onChange(of: mapRegion) { newRegion in
                onViewportChanged(newRegion)
            }

            // Show missing partition prompt if needed
            if !missingPartitions.isEmpty {
                VStack {
                    Spacer()
                    MissingPartitionsPrompt(
                        partitions: missingPartitions,
                        onDownload: downloadMissingPartitions
                    )
                }
            }
        }
        .onAppear {
            loadInitialPartitions()
        }
    }

    private func onViewportChanged(_ viewport: MKCoordinateRegion) {
        // Load partitions for new viewport
        Task {
            await lazyLoader.loadPartitionsForViewport(
                region: region,
                viewport: viewport
            )

            // Check for missing partitions
            missingPartitions = lazyLoader.getMissingPartitions(
                region: region,
                viewport: viewport
            )

            // Query climbs from loaded partitions
            climbs = dbManager.queryClimbs(in: viewport)
        }
    }
}

struct MissingPartitionsPrompt: View {
    let partitions: [Partition]
    let onDownload: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Text("Some areas in view aren't downloaded")
                .font(.subheadline)
                .fontWeight(.medium)

            Text(partitions.map { $0.displayName }.joined(separator: ", "))
                .font(.caption)
                .foregroundColor(.secondary)

            Button("Download", action: onDownload)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
        .padding()
    }
}
```

---

## Verification Checklist

After implementing, verify:

1. **Partition Detection**: App correctly identifies `is_partitioned: true` regions from index.json
2. **Bounds Display**: Partition list shows bounds info (if available) and climb counts
3. **Individual Download**: Can download a single partition (e.g., just SoCal)
4. **Checksum Verification**: SHA256 verified after download (check logs)
5. **Lazy Loading**: Only loads partitions visible in current viewport
6. **Cross-Partition Queries**: Climbs from multiple loaded partitions appear on map
7. **Storage Management**: Can delete individual partitions to free space
8. **Error Handling**: Graceful handling of download failures, checksum mismatches

---

## File Structure Summary

```
YourApp/
├── Models/
│   ├── Partition.swift              # Partition, PartitionBounds
│   ├── RegionMetadata.swift         # Updated with partition support
│   ├── DownloadedPartition.swift    # Track downloaded partitions
│   └── Climb.swift                  # Your existing climb model
├── Services/
│   ├── IndexService.swift           # Fetch and parse index.json
│   ├── PartitionDownloadManager.swift
│   ├── PartitionedDatabaseManager.swift
│   └── LazyPartitionLoader.swift
├── Views/
│   ├── RegionListView.swift         # List of available regions
│   ├── RegionDetailView.swift       # Updated to show partitions
│   ├── PartitionPickerView.swift    # New - select partitions
│   ├── ClimbMapView.swift           # Updated with lazy loading
│   └── ClimbDetailView.swift
└── Utilities/
    └── Extensions.swift             # URLSession download extension
```
