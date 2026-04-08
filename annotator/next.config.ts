import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "fsn1.your-objectstorage.com",
        pathname: "/strata-training-data/annotations/**",
      },
    ],
  },
};

export default nextConfig;
