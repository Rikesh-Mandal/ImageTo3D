import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ImageTo3D | Monocular 3D Reconstruction",
  description:
    "Estimate depth and generate a coloured 3D mesh from a single RGB image.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
