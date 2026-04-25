'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { useSettings, DEFAULT_SETTINGS } from '@/lib/settings';
import { Settings as SettingsIcon, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function SettingsPage() {
  const [settings, update] = useSettings();

  const reset = () => update(DEFAULT_SETTINGS);

  return (
    <div className="p-8 max-w-3xl">
      <div className="mb-8 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center gap-2">
            <SettingsIcon className="h-7 w-7" /> Settings
          </h1>
          <p className="text-gray-600">
            Display and data-loading preferences. Stored locally in your browser.
          </p>
        </div>
        <Button type="button" variant="outline" onClick={reset}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset to defaults
        </Button>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Units</CardTitle>
            <CardDescription>
              Affects distance, elevation, and chart axes throughout the app.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              {(['imperial', 'metric'] as const).map(u => (
                <button
                  key={u}
                  type="button"
                  onClick={() => update({ units: u })}
                  className={`rounded-lg border px-4 py-3 text-left transition-colors ${
                    settings.units === u
                      ? 'border-blue-600 bg-blue-50 ring-1 ring-blue-600'
                      : 'border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  <div className="font-semibold capitalize">{u}</div>
                  <div className="text-xs text-gray-500">
                    {u === 'imperial' ? 'miles, feet' : 'kilometers, meters'}
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Climb load limit</CardTitle>
            <CardDescription>
              Cap the number of climbs loaded from each file to keep the UI responsive on large
              regions. Top climbs are kept (sorted by score).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={settings.climbLimitEnabled}
                onChange={(e) => update({ climbLimitEnabled: e.target.checked })}
                className="w-4 h-4 rounded"
              />
              <span className="text-sm">Limit climbs per loaded file</span>
            </label>

            <div className={settings.climbLimitEnabled ? '' : 'opacity-50 pointer-events-none'}>
              <Label className="text-sm">Max climbs per file: {settings.maxClimbsToLoad.toLocaleString()}</Label>
              <Slider
                value={[settings.maxClimbsToLoad]}
                onValueChange={(v) => update({ maxClimbsToLoad: v[0] })}
                min={100}
                max={10000}
                step={100}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>100</span>
                <span>10,000</span>
              </div>
            </div>

            {!settings.climbLimitEnabled && (
              <div className="rounded-md bg-amber-50 border border-amber-200 p-3 text-xs text-amber-800">
                Disabling the limit lets large regions consume significant memory. Increase your
                browser tab memory budget if you see slow rendering.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>About</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600 space-y-1">
            <div>Climb Analyzer Web UI</div>
            <div>
              Data:{' '}
              <a
                className="text-blue-600 hover:underline"
                href="https://github.com/stevehollx/global-road-and-trail-climbs"
                target="_blank"
                rel="noopener noreferrer"
              >
                stevehollx/global-road-and-trail-climbs
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
