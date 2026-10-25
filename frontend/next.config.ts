import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  transpilePackages: ["three"],

  async rewrites() {
    return {
      beforeFiles: [
        {
          source: "/api/:path*",
          destination: "http://127.0.0.1:8001/api/:path*",
        },
      ],
    };
  },
};

export default nextConfig;
