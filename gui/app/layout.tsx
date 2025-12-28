import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { AlertTriangle } from "lucide-react";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Climb Analyzer - Web GUI",
  description: "Analyze and visualize road climbs from OpenStreetMap",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased`}>
        <div className="flex h-screen overflow-hidden flex-col">
          {/* Beta Warning Banner */}
          <div className="bg-yellow-500 text-gray-900 px-4 py-2 flex items-center justify-center gap-2 text-sm font-medium shadow-md z-50">
            <AlertTriangle className="h-4 w-4" />
            <span>
              GUI v2.0 - Recommended: Run analysis via CLI and visualize results in GUI due to limited testing of GUI analysis features.
            </span>
          </div>

          <div className="flex flex-1 overflow-hidden">
            <Sidebar />
            <main className="flex-1 overflow-y-auto bg-gray-50">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
