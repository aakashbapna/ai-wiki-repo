import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      colors: {
        ink: "#0F172A",
        mist: "#E2E8F0",
        cloud: "#F8FAFC",
        accent: "#14B8A6",
        accentDark: "#0F766E"
      },
      boxShadow: {
        panel: "0 20px 40px -30px rgba(15, 23, 42, 0.6)",
      },
    },
  },
  plugins: [],
};

export default config;
