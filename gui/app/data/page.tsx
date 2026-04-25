'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Cloud,
  CheckCircle2,
  Download,
  ChevronRight,
  ChevronDown,
  FileSpreadsheet,
  Database,
  RefreshCw,
  ExternalLink,
  Loader2,
  AlertTriangle,
} from 'lucide-react';

interface RegionMeta {
  region_name: string;
  version: string;
  release_tag: string;
  release_url: string;
  climb_count: number | null;
  files: string[];
  download_urls: string[];
  file_sizes: number[];
  total_size: number;
  database_file: string | null;
  database_size: number | null;
  database_url: string | null;
  is_partitioned?: boolean;
  partitions?: Array<{
    partition_id: string;
    display_name: string;
    database_file: string;
    database_size: number;
    database_url: string;
    climb_count?: number | null;
  }> | null;
  published_at: string;
}

interface IndexJson {
  source?: 'local' | 'remote';
  generated_at: string;
  summary: {
    total_regions: number;
    total_climbs: number;
    total_size_mb: number;
  };
  regions: Record<string, RegionMeta>;
}

type TreeNode = {
  name: string;
  fullPath: string;
  children: Map<string, TreeNode>;
  region?: { path: string; meta: RegionMeta };
};

function buildTree(regions: Record<string, RegionMeta>): TreeNode {
  const root: TreeNode = { name: '', fullPath: '', children: new Map() };
  for (const [path, meta] of Object.entries(regions)) {
    const parts = path.split('/');
    let cursor = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      if (!cursor.children.has(part)) {
        cursor.children.set(part, {
          name: part,
          fullPath: parts.slice(0, i + 1).join('/'),
          children: new Map(),
        });
      }
      cursor = cursor.children.get(part)!;
    }
    cursor.region = { path, meta };
  }
  return root;
}

// Title-case but keep small connector words lowercase so "united-states-of-america"
// becomes "United States of America" instead of "United States Of America".
const SMALL_WORDS = new Set(['of', 'and', 'the', 'in', 'on', 'for', 'a', 'an', 'to']);
function prettyName(slug: string): string {
  return slug
    .split('-')
    .map((s, i) => {
      const lower = s.toLowerCase();
      if (i > 0 && SMALL_WORDS.has(lower)) return lower;
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(' ');
}

function formatBytes(b: number | null | undefined): string {
  if (!b) return '—';
  if (b > 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b > 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b > 1e3) return `${(b / 1e3).toFixed(1)} KB`;
  return `${b} B`;
}

function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—';
  return n.toLocaleString();
}

export default function DataPage() {
  const [index, setIndex] = useState<IndexJson | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [downloaded, setDownloaded] = useState<Set<string>>(new Set());
  const [downloading, setDownloading] = useState<Record<string, 'pending' | 'error'>>({});

  const loadIndex = async (forceRemote = false) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/repo-index${forceRemote ? '?remote=1' : ''}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
      setIndex(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const loadDownloaded = async () => {
    try {
      const res = await fetch('/api/output-files');
      if (!res.ok) return;
      const data = await res.json();
      const set = new Set<string>((data.outputFiles || []).map((f: { filename: string }) => f.filename));
      setDownloaded(set);
    } catch {}
  };

  useEffect(() => {
    loadIndex();
    loadDownloaded();
  }, []);

  const tree = useMemo(() => (index ? buildTree(index.regions) : null), [index]);

  const toggle = (path: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const download = async (url: string, filename: string) => {
    setDownloading(prev => ({ ...prev, [filename]: 'pending' }));
    try {
      const res = await fetch('/api/data-download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, filename }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
      await loadDownloaded();
      setDownloading(prev => {
        const next = { ...prev };
        delete next[filename];
        return next;
      });
    } catch (e) {
      console.error(e);
      setDownloading(prev => ({ ...prev, [filename]: 'error' }));
    }
  };

  const matchesSearch = (node: TreeNode): boolean => {
    if (!search) return true;
    const q = search.toLowerCase();
    if (node.name.toLowerCase().includes(q)) return true;
    if (node.fullPath.toLowerCase().includes(q)) return true;
    if (node.region?.meta.region_name.toLowerCase().includes(q)) return true;
    for (const child of node.children.values()) {
      if (matchesSearch(child)) return true;
    }
    return false;
  };

  // Auto-expand all nodes that match the search
  useEffect(() => {
    if (!search || !tree) return;
    const next = new Set<string>();
    const walk = (n: TreeNode) => {
      for (const c of n.children.values()) {
        if (matchesSearch(c)) {
          next.add(c.fullPath);
          walk(c);
        }
      }
    };
    walk(tree);
    setExpanded(next);
  }, [search, tree]);

  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center gap-2">
            <Cloud className="h-7 w-7" />
            Data Library
          </h1>
          <p className="text-gray-600">
            Browse and download pre-built climb databases from{' '}
            <a
              className="text-blue-600 hover:underline"
              href="https://github.com/stevehollx/global-road-and-trail-climbs"
              target="_blank"
              rel="noopener noreferrer"
            >
              global-road-and-trail-climbs
            </a>
            . Files are saved into <code>output/</code> and become available on the Visualize page.
          </p>
        </div>
        <Button variant="outline" onClick={() => { loadIndex(true); loadDownloaded(); }}>
          <RefreshCw className="h-4 w-4 mr-2" /> Refresh
        </Button>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-gray-500 py-8">
          <Loader2 className="h-5 w-5 animate-spin" /> Loading index…
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-800 text-sm flex items-start gap-2">
          <AlertTriangle className="h-5 w-5 flex-shrink-0" />
          <div>
            <div className="font-semibold">Couldn&apos;t load index.json</div>
            <div className="mt-1 text-red-700">{error}</div>
          </div>
        </div>
      )}

      {index && tree && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
            <SummaryCard label="Regions" value={formatCount(index.summary.total_regions)} />
            <SummaryCard label="Total climbs" value={formatCount(index.summary.total_climbs)} />
            <SummaryCard label="Total size" value={`${index.summary.total_size_mb.toLocaleString()} MB`} />
          </div>

          <div className="mb-4">
            <Input
              placeholder="Search regions, countries, continents…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Available regions</CardTitle>
              <CardDescription>
                Click a region to expand its files. Source: {index.source === 'local' ? 'local index.json' : 'github.com'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-1">
              {Array.from(tree.children.values())
                .filter(matchesSearch)
                .sort((a, b) => a.name.localeCompare(b.name))
                .map(node => (
                  <TreeRow
                    key={node.fullPath}
                    node={node}
                    depth={0}
                    expanded={expanded}
                    onToggle={toggle}
                    matchesSearch={matchesSearch}
                    downloaded={downloaded}
                    downloading={downloading}
                    onDownload={download}
                  />
                ))}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-bold text-gray-900 mt-1">{value}</div>
    </div>
  );
}

function TreeRow(props: {
  node: TreeNode;
  depth: number;
  expanded: Set<string>;
  onToggle: (p: string) => void;
  matchesSearch: (n: TreeNode) => boolean;
  downloaded: Set<string>;
  downloading: Record<string, 'pending' | 'error'>;
  onDownload: (url: string, filename: string) => void;
}) {
  const { node, depth, expanded, onToggle, matchesSearch, downloaded, downloading, onDownload } = props;
  const isOpen = expanded.has(node.fullPath);
  const hasChildren = node.children.size > 0;
  const isRegion = !!node.region;

  return (
    <div>
      <div
        className={`flex items-start gap-2 py-1.5 px-2 rounded hover:bg-gray-50 cursor-pointer`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => (hasChildren || isRegion) && onToggle(node.fullPath)}
      >
        {hasChildren || isRegion ? (
          isOpen ? <ChevronDown className="h-4 w-4 mt-0.5 text-gray-400" />
                : <ChevronRight className="h-4 w-4 mt-0.5 text-gray-400" />
        ) : <span className="w-4" />}
        <span className={`font-${isRegion ? 'medium' : 'semibold'} text-sm`}>
          {prettyName(node.name)}
        </span>
        {isRegion && node.region && (
          <span className="text-xs text-gray-500 ml-2">
            {formatCount(node.region.meta.climb_count)} climbs · {formatBytes(node.region.meta.total_size + (node.region.meta.database_size || 0))}
          </span>
        )}
      </div>

      {isOpen && isRegion && node.region && (
        <RegionFiles
          path={node.region.path}
          meta={node.region.meta}
          depth={depth + 1}
          downloaded={downloaded}
          downloading={downloading}
          onDownload={onDownload}
        />
      )}
      {isOpen && hasChildren && (
        Array.from(node.children.values())
          .filter(matchesSearch)
          .sort((a, b) => a.name.localeCompare(b.name))
          .map(child => (
            <TreeRow
              key={child.fullPath}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              onToggle={onToggle}
              matchesSearch={matchesSearch}
              downloaded={downloaded}
              downloading={downloading}
              onDownload={onDownload}
            />
          ))
      )}
    </div>
  );
}

function RegionFiles({
  path,
  meta,
  depth,
  downloaded,
  downloading,
  onDownload,
}: {
  path: string;
  meta: RegionMeta;
  depth: number;
  downloaded: Set<string>;
  downloading: Record<string, 'pending' | 'error'>;
  onDownload: (url: string, filename: string) => void;
}) {
  type FileRow = { filename: string; url: string; size: number; kind: 'xlsx' | 'sqlite' };
  const files: FileRow[] = [];
  meta.files.forEach((name, i) => {
    files.push({ filename: name, url: meta.download_urls[i], size: meta.file_sizes[i], kind: 'xlsx' });
  });
  if (meta.database_file && meta.database_url) {
    files.push({ filename: meta.database_file, url: meta.database_url, size: meta.database_size || 0, kind: 'sqlite' });
  }
  if (meta.is_partitioned && meta.partitions) {
    for (const p of meta.partitions) {
      files.push({ filename: p.database_file, url: p.database_url, size: p.database_size, kind: 'sqlite' });
    }
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50/60 mx-2 my-2 p-3 space-y-2" style={{ marginLeft: `${depth * 16 + 24}px` }}>
      <div className="flex items-center justify-between text-xs text-gray-600">
        <div>
          <div className="font-semibold text-gray-800">{meta.region_name} · v{meta.version}</div>
          <div>Path: <code className="bg-gray-100 px-1 py-0.5 rounded">{path}</code></div>
        </div>
        <a
          href={meta.release_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline flex items-center gap-1"
        >
          GitHub release <ExternalLink className="h-3 w-3" />
        </a>
      </div>

      <div className="space-y-1">
        {files.map(f => {
          const have = downloaded.has(f.filename);
          const state = downloading[f.filename];
          return (
            <div key={f.filename} className="flex items-center gap-2 bg-white border border-gray-200 rounded p-2 text-sm">
              {f.kind === 'sqlite'
                ? <Database className="h-4 w-4 text-purple-600 flex-shrink-0" />
                : <FileSpreadsheet className="h-4 w-4 text-green-600 flex-shrink-0" />}
              <div className="flex-1 min-w-0">
                <div className="font-mono text-xs truncate">{f.filename}</div>
                <div className="text-xs text-gray-500">{formatBytes(f.size)}</div>
              </div>
              {have ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-green-700 flex items-center gap-1">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Downloaded
                  </span>
                  <Link
                    href={`/visualize?file=${encodeURIComponent(f.filename)}`}
                    className="text-xs px-2 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded border border-blue-200"
                  >
                    Open
                  </Link>
                </div>
              ) : state === 'pending' ? (
                <span className="text-xs text-gray-500 flex items-center gap-1">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> Downloading…
                </span>
              ) : state === 'error' ? (
                <Button size="sm" variant="outline" onClick={() => onDownload(f.url, f.filename)}>
                  <AlertTriangle className="h-3.5 w-3.5 mr-1 text-red-600" /> Retry
                </Button>
              ) : (
                <Button size="sm" variant="outline" onClick={() => onDownload(f.url, f.filename)}>
                  <Download className="h-3.5 w-3.5 mr-1" /> Download
                </Button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
