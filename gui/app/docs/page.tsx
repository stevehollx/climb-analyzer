import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BookOpen, ExternalLink } from 'lucide-react';

export default function DocsPage() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-2">Documentation</h1>
      <p className="text-gray-600 mb-8">
        Learn how to use the Climb Analyzer
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Quick Start Guide</CardTitle>
            <CardDescription>Get up and running quickly</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm">
              Learn how to configure your deployment mode, download data, and
              run your first analysis.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Analysis Modes</CardTitle>
            <CardDescription>Understand the different analysis types</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <ul className="text-sm space-y-1">
              <li>• <strong>Address Mode:</strong> Analyze climbs within a radius</li>
              <li>• <strong>Region Mode:</strong> Analyze an entire state/country</li>
              <li>• <strong>Batch Mode:</strong> Process multiple regions</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Scoring Algorithms</CardTitle>
            <CardDescription>Different methods to rate climbs</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <ul className="text-sm space-y-1">
              <li>• <strong>Basic:</strong> grade × distance</li>
              <li>• <strong>FIETS:</strong> gradient-based (Fietsklim formula)</li>
              <li>• <strong>PDI:</strong> comprehensive difficulty index</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>CLI Documentation</CardTitle>
            <CardDescription>Command-line usage reference</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm mb-2">
              View the complete CLI documentation for advanced usage and
              automation.
            </p>
            <a
              href="#"
              className="text-sm text-blue-600 hover:underline inline-flex items-center gap-1"
            >
              View CLI Docs
              <ExternalLink className="h-3 w-3" />
            </a>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            <CardTitle>Example Workflows</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">1. Quick Local Analysis</h4>
            <ol className="text-sm space-y-1 list-decimal list-inside text-gray-600">
              <li>Go to Run Analysis</li>
              <li>Select "Address (Radius)" mode</li>
              <li>Enter your city and a 25-mile radius</li>
              <li>Choose your preferred units and scoring</li>
              <li>Click "Start Analysis"</li>
              <li>View results in Visualize Climbs</li>
            </ol>
          </div>

          <div>
            <h4 className="font-semibold mb-2">2. Full State Analysis</h4>
            <ol className="text-sm space-y-1 list-decimal list-inside text-gray-600">
              <li>Optionally pre-download data in Download Data page</li>
              <li>Go to Run Analysis</li>
              <li>Select "Single Region" mode</li>
              <li>Enter state name (e.g., "Colorado" or "CO")</li>
              <li>Configure surface filters and scoring</li>
              <li>Start the analysis (may take hours for large states)</li>
            </ol>
          </div>

          <div>
            <h4 className="font-semibold mb-2">3. Batch Processing</h4>
            <ol className="text-sm space-y-1 list-decimal list-inside text-gray-600">
              <li>Go to Run Analysis</li>
              <li>Select "Batch Regions" mode</li>
              <li>Enter comma-separated regions (e.g., "VT, NH, ME")</li>
              <li>Enable "Delete Data After Analysis" to save space</li>
              <li>Start batch processing</li>
            </ol>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
