import type { NextConfig } from "next";

const fastApiUrl = (process.env.FASTAPI_URL ?? "http://127.0.0.1:8001").replace(
  /\/$/,
  "",
);

const nextConfig: NextConfig = {
  output: "standalone",

  transpilePackages: ["three"],

  async rewrites() {
    return {
      beforeFiles: [
        {
          source: "/api/:path*",
          destination: `${fastApiUrl}/api/:path*`,
        },
      ],
    };
  },
};

export default nextConfig;
