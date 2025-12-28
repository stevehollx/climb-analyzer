'use client';

import { useState, useCallback } from 'react';
import { MapPin, Navigation, X, Loader2 } from 'lucide-react';

interface LocationFilterProps {
  onLocationChange: (location: { lat: number; lon: number; radius: number } | null) => void;
  initialRadius?: number;
}

export function LocationFilter({ onLocationChange, initialRadius = 25 }: LocationFilterProps) {
  const [address, setAddress] = useState('');
  const [lat, setLat] = useState<string>('');
  const [lon, setLon] = useState<string>('');
  const [radius, setRadius] = useState(initialRadius);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeLocation, setActiveLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [inputMode, setInputMode] = useState<'address' | 'coords'>('address');

  // Geocode address using Nominatim API
  const geocodeAddress = useCallback(async (addressText: string) => {
    if (!addressText.trim()) {
      setError('Please enter an address');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(addressText)}&limit=1`,
        {
          headers: {
            'User-Agent': 'ClimbAnalyzer/1.0',
          },
        }
      );

      if (!response.ok) {
        throw new Error('Geocoding service unavailable');
      }

      const data = await response.json();

      if (data.length === 0) {
        setError('Address not found');
        return;
      }

      const location = {
        lat: parseFloat(data[0].lat),
        lon: parseFloat(data[0].lon),
      };

      setActiveLocation(location);
      onLocationChange({ ...location, radius });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Geocoding failed');
    } finally {
      setIsLoading(false);
    }
  }, [radius, onLocationChange]);

  // Apply manual coordinates
  const applyCoordinates = useCallback(() => {
    const latNum = parseFloat(lat);
    const lonNum = parseFloat(lon);

    if (isNaN(latNum) || isNaN(lonNum)) {
      setError('Please enter valid coordinates');
      return;
    }

    if (latNum < -90 || latNum > 90) {
      setError('Latitude must be between -90 and 90');
      return;
    }

    if (lonNum < -180 || lonNum > 180) {
      setError('Longitude must be between -180 and 180');
      return;
    }

    setError(null);
    const location = { lat: latNum, lon: lonNum };
    setActiveLocation(location);
    onLocationChange({ ...location, radius });
  }, [lat, lon, radius, onLocationChange]);

  // Use browser geolocation ("Near Me")
  const useMyLocation = useCallback(() => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by your browser');
      return;
    }

    setIsLoading(true);
    setError(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const location = {
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        };
        setActiveLocation(location);
        setLat(location.lat.toFixed(6));
        setLon(location.lon.toFixed(6));
        onLocationChange({ ...location, radius });
        setIsLoading(false);
      },
      (err) => {
        let message = 'Unable to get your location';
        if (err.code === err.PERMISSION_DENIED) {
          message = 'Location permission denied';
        } else if (err.code === err.POSITION_UNAVAILABLE) {
          message = 'Location information unavailable';
        } else if (err.code === err.TIMEOUT) {
          message = 'Location request timed out';
        }
        setError(message);
        setIsLoading(false);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0,
      }
    );
  }, [radius, onLocationChange]);

  // Clear location filter
  const clearLocation = useCallback(() => {
    setActiveLocation(null);
    setAddress('');
    setLat('');
    setLon('');
    setError(null);
    onLocationChange(null);
  }, [onLocationChange]);

  // Update radius and re-apply filter if location is set
  const handleRadiusChange = useCallback((newRadius: number) => {
    setRadius(newRadius);
    if (activeLocation) {
      onLocationChange({ ...activeLocation, radius: newRadius });
    }
  }, [activeLocation, onLocationChange]);

  return (
    <div className="space-y-3">
      {/* Input Mode Toggle */}
      <div className="flex items-center gap-2 text-sm">
        <button
          onClick={() => setInputMode('address')}
          className={`px-2 py-1 rounded ${
            inputMode === 'address'
              ? 'bg-blue-100 text-blue-700'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Address
        </button>
        <button
          onClick={() => setInputMode('coords')}
          className={`px-2 py-1 rounded ${
            inputMode === 'coords'
              ? 'bg-blue-100 text-blue-700'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Coordinates
        </button>
      </div>

      {/* Address Input */}
      {inputMode === 'address' && (
        <div className="flex gap-2">
          <input
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="Enter address..."
            className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                geocodeAddress(address);
              }
            }}
          />
          <button
            onClick={() => geocodeAddress(address)}
            disabled={isLoading || !address.trim()}
            className="px-3 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-1"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <MapPin className="h-4 w-4" />
            )}
          </button>
        </div>
      )}

      {/* Coordinates Input */}
      {inputMode === 'coords' && (
        <div className="flex gap-2">
          <input
            type="text"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            placeholder="Latitude"
            className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            type="text"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            placeholder="Longitude"
            className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={applyCoordinates}
            disabled={isLoading || !lat || !lon}
            className="px-3 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-1"
          >
            <MapPin className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Near Me Button */}
      <button
        onClick={useMyLocation}
        disabled={isLoading}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Navigation className="h-4 w-4" />
        )}
        Near Me
      </button>

      {/* Radius Slider */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <label className="text-gray-700 font-medium">Radius</label>
          <span className="text-gray-600">{radius} miles</span>
        </div>
        <input
          type="range"
          min={1}
          max={100}
          value={radius}
          onChange={(e) => handleRadiusChange(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>1 mi</span>
          <span>100 mi</span>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-2 p-2 bg-red-50 text-red-700 rounded-lg text-sm">
          <X className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Active Location Display */}
      {activeLocation && (
        <div className="flex items-center justify-between p-2 bg-blue-50 rounded-lg text-sm">
          <div className="flex items-center gap-2 text-blue-700">
            <MapPin className="h-4 w-4" />
            <span>
              {activeLocation.lat.toFixed(4)}°, {activeLocation.lon.toFixed(4)}°
            </span>
          </div>
          <button
            onClick={clearLocation}
            className="text-blue-600 hover:text-blue-800"
            title="Clear location filter"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}
