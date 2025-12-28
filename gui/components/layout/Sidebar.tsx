'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Home,
  Play,
  Map,
  Download,
  Settings,
  Trash2,
  BookOpen,
  Mountain
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Download Data', href: '/download', icon: Download },
  { name: 'Analyze Climbs', href: '/analyze', icon: Play },
  { name: 'Visualize Climbs', href: '/visualize', icon: Map },
  { name: 'Delete Data', href: '/manage', icon: Trash2 },
  { name: 'Documentation', href: '/docs', icon: BookOpen },
  { name: 'Configuration', href: '/config', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-screen w-64 flex-col bg-gray-900 text-white">
      {/* Logo */}
      <Link href="/" className="flex h-16 items-center gap-2 px-6 border-b border-gray-800 hover:bg-gray-800 transition-colors">
        <Mountain className="h-8 w-8 text-blue-500" />
        <div>
          <h1 className="text-lg font-bold">Climb Analyzer</h1>
          <p className="text-xs text-gray-400">Web UI</p>
        </div>
      </Link>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-gray-800 p-4">
        <p className="text-xs text-gray-500">
          v1.0.0
        </p>
      </div>
    </div>
  );
}
