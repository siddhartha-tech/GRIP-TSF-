import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bass: {
          gold: "#C5A028",
          "gold-light": "#E8D374",
          "gold-dark": "#8B7118",
          green: "#2D5016",
          "green-light": "#4A7A2E",
          "green-dark": "#1A3009",
          brown: "#5C3D1E",
          cream: "#FFF8E7",
          olive: "#6B7B3A",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
