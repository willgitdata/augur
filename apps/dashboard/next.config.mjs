/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    QUERYBRAIN_URL: process.env.QUERYBRAIN_URL ?? "http://localhost:3001",
  },
};
export default nextConfig;
