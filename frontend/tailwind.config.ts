import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f5f6fa",
          100: "#e8eaf3",
          200: "#c8cce0",
          300: "#9aa1c2",
          400: "#6e76a0",
          500: "#4a5184",
          600: "#363c6a",
          700: "#272c52",
          800: "#191c39",
          900: "#0d1027",
          950: "#070918",
        },
        accent: {
          400: "#7ee0ff",
          500: "#3ec5ff",
          600: "#1ea0e8",
        },
        flag: "#ff5577",
        warn: "#ffb547",
        ok: "#5fd28a",
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(126,224,255,0.25), 0 12px 60px -10px rgba(62,197,255,0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
