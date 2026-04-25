'use client';

import { useEffect, useState } from 'react';
import { Copy, Check, Terminal, Info } from 'lucide-react';
import { AnalysisConfig } from '@/types/climb';
import { buildAnalyzeCommand } from '@/lib/cli-command';

interface CommandPreviewProps {
  config: AnalysisConfig;
  cloudUploadEnabled?: boolean;
}

export function CommandPreview({ config, cloudUploadEnabled }: CommandPreviewProps) {
  const command = buildAnalyzeCommand(config, { cloudUploadEnabled });
  const [copied, setCopied] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(t);
  }, [copied]);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = command;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand('copy');
        setCopied(true);
      } finally {
        document.body.removeChild(ta);
      }
    }
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-900 text-gray-100">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <div className="flex items-center gap-2 text-sm font-medium text-gray-300">
          <Terminal className="h-4 w-4" />
          Equivalent CLI command
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setShowHelp((v) => !v)}
            className="p-1.5 rounded hover:bg-gray-800 text-gray-400 hover:text-gray-200"
            title="How to run this in your terminal"
          >
            <Info className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={onCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-100 border border-gray-700"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-green-400" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                Copy
              </>
            )}
          </button>
        </div>
      </div>

      <button
        type="button"
        onClick={onCopy}
        className="block w-full text-left px-3 py-3 font-mono text-xs leading-relaxed overflow-x-auto whitespace-pre-wrap break-all hover:bg-gray-800/40 transition-colors"
        title="Click to copy"
      >
        <span className="text-gray-500 select-none">$ </span>
        <span>{command}</span>
      </button>

      {showHelp && (
        <div className="px-3 py-3 border-t border-gray-800 text-xs text-gray-300 space-y-2">
          <p className="font-medium text-gray-200">To run this:</p>
          <ol className="list-decimal list-inside space-y-1 text-gray-400">
            <li>Click the command above (or the Copy button) to copy it.</li>
            <li>Open your terminal in the project root (<code className="text-gray-200">/mnt/usb1/ca11</code>).</li>
            <li>Paste and run. Inside the docker container if applicable: <code className="text-gray-200">docker compose exec climb-analyzer ...</code></li>
          </ol>
          <p className="text-gray-500 pt-1">
            Browsers can&apos;t launch a terminal directly. If you want one-click execution, the
            existing &quot;Run in Background&quot; button below spawns the same command via the
            Next.js server process.
          </p>
        </div>
      )}
    </div>
  );
}
