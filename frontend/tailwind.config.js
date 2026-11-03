/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: '#f7f6f3',
        card: '#ffffff',
        line: '#e8e5de',
        soft: '#eeece7',
        ink: '#17161a',
        muted: '#6b6a66',
        faint: '#8a8781',
        accent: '#b4442e',
        quiet: '#3f6b52',
        busy: '#a1731f',
        crowded: '#b4442e',
      },
    },
  },
  plugins: [],
}
