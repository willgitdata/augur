import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        ink: {
          900: "#0a0a0a",
          800: "#171717",
          700: "#262626",
          500: "#737373",
          300: "#d4d4d4",
        },
        accent: {
          500: "#10b981",
          600: "#059669",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
