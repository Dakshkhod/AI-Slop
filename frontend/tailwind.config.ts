import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Neutral grayscale — pure black at the bottom
        ink: {
          50:  "#fafafa",
          100: "#e5e5e5",
          200: "#bdbdbd",
          300: "#8a8a8a",
          400: "#5e5e5e",
          500: "#3f3f3f",
          600: "#272727",
          700: "#1a1a1a",
          800: "#111111",
          900: "#0a0a0a",
          950: "#000000",
        },
        // Subtle near-white accent — used sparingly
        accent: {
          400: "#f5f5f5",
          500: "#e5e5e5",
          600: "#a3a3a3",
        },
        // Desaturated status colors
        flag: "#e2545b",
        warn: "#d6a14a",
        ok:   "#5fae7e",
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.06), 0 12px 40px -10px rgba(0,0,0,0.6)",
      },
    },
  },
  plugins: [],
};

export default config;
