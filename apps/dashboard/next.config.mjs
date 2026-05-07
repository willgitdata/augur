/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    AUGUR_URL: process.env.AUGUR_URL ?? "http://localhost:3001",
  },
};
export default nextConfig;
