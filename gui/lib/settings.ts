'use client';

import { useEffect, useState } from 'react';

export type Units = 'imperial' | 'metric';

export interface AppSettings {
  units: Units;
  climbLimitEnabled: boolean;
  maxClimbsToLoad: number;
}

export const DEFAULT_SETTINGS: AppSettings = {
  units: 'imperial',
  climbLimitEnabled: true,
  maxClimbsToLoad: 1000,
};

const STORAGE_KEY = 'climbAnalyzer.appSettings.v1';
const EVENT_NAME = 'climbAnalyzer.appSettings.changed';

export function readSettings(): AppSettings {
  if (typeof window === 'undefined') return DEFAULT_SETTINGS;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function writeSettings(next: Partial<AppSettings>): AppSettings {
  const merged = { ...readSettings(), ...next };
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
    window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: merged }));
  }
  return merged;
}

export function useSettings(): [AppSettings, (next: Partial<AppSettings>) => void] {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    setSettings(readSettings());
    const onChange = (e: Event) => {
      const ce = e as CustomEvent<AppSettings>;
      if (ce.detail) setSettings(ce.detail);
      else setSettings(readSettings());
    };
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setSettings(readSettings());
    };
    window.addEventListener(EVENT_NAME, onChange as EventListener);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener(EVENT_NAME, onChange as EventListener);
      window.removeEventListener('storage', onStorage);
    };
  }, []);

  const update = (next: Partial<AppSettings>) => setSettings(writeSettings(next));
  return [settings, update];
}

// Unit conversion helpers. Source data is stored in imperial (miles, feet).
export const FT_PER_M = 3.28084;
export const MI_PER_KM = 0.621371;

export function distanceFromMiles(miles: number, units: Units): { value: number; unit: string } {
  return units === 'imperial'
    ? { value: miles, unit: 'mi' }
    : { value: miles / MI_PER_KM, unit: 'km' };
}

export function elevationFromFeet(feet: number, units: Units): { value: number; unit: string } {
  return units === 'imperial'
    ? { value: feet, unit: 'ft' }
    : { value: feet / FT_PER_M, unit: 'm' };
}

export function fmtDistance(miles: number | undefined, units: Units, digits = 2): string {
  if (miles === undefined || miles === null || isNaN(miles)) return '—';
  const { value, unit } = distanceFromMiles(miles, units);
  return `${value.toFixed(digits)} ${unit}`;
}

export function fmtElevation(feet: number | undefined, units: Units): string {
  if (feet === undefined || feet === null || isNaN(feet)) return '—';
  const { value, unit } = elevationFromFeet(feet, units);
  return `${Math.round(value).toLocaleString()} ${unit}`;
}
