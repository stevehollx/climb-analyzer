import { AnalysisConfig } from '@/types/climb';

export interface BuildCommandOptions {
  cloudUploadEnabled?: boolean;
  scriptName?: string;
}

function shellQuote(value: string): string {
  if (value === '' || /[\s"'$`\\!*?{}()<>|&;~#]/.test(value)) {
    return `'${value.replace(/'/g, `'\\''`)}'`;
  }
  return value;
}

export function buildAnalyzeArgs(
  config: AnalysisConfig,
  options: BuildCommandOptions = {}
): string[] {
  const args: string[] = [];

  if (config.mode === 'address') {
    if (config.address) args.push('--address', config.address);
    if (config.radius !== undefined) args.push('--distance', String(config.radius));
  } else if (config.mode === 'region') {
    const list = config.regions && config.regions.length > 0
      ? config.regions.join(',')
      : config.region;
    if (list) args.push('--run-region', list);
  } else if (config.mode === 'batch') {
    if (config.regions && config.regions.length > 0) {
      args.push('--run-region', config.regions.join(','));
    }
  }

  if (config.surfaceFilter && config.surfaceFilter !== 'all') {
    args.push('--surface-filter', config.surfaceFilter);
  }

  if (config.units) {
    args.push('--units', config.units);
  }

  if (config.minScore !== undefined && config.minScore !== null) {
    args.push('--score-type', 'basic');
    args.push('--min-score', String(config.minScore));
  }

  if (config.deleteDataOnComplete) {
    args.push('--cleanup-all-data');
  }

  if (options.cloudUploadEnabled === false) {
    args.push('--no-cloud-upload');
  }

  return args;
}

export function buildAnalyzeCommand(
  config: AnalysisConfig,
  options: BuildCommandOptions = {}
): string {
  const script = options.scriptName || 'climb_analyzer.py';
  const args = buildAnalyzeArgs(config, options);
  return ['python3', script, ...args].map(shellQuote).join(' ');
}
