/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        ink: {
          DEFAULT: 'var(--ink)',
          2: 'var(--ink-2)',
          3: 'var(--ink-3)',
          4: 'var(--ink-4)',
          5: 'var(--ink-5)',
          6: 'var(--ink-6)',
        },
        line: {
          DEFAULT: 'var(--line)',
          2: 'var(--line-2)',
          3: 'var(--line-3)',
        },
        accent: 'var(--accent)',
        brand: {
          blue: 'var(--blue)',
          red: 'var(--red)',
          amber: 'var(--amber)',
          green: 'var(--green)',
        }
      },
      borderRadius: {
        'lg': 'var(--r-lg)',
        DEFAULT: 'var(--r)',
        'sm': 'var(--r-sm)',
        'xs': 'var(--r-xs)',
      },
      boxShadow: {
        '1': 'var(--sh-1)',
        '2': 'var(--sh-2)',
        '3': 'var(--sh-3)',
      },
      fontFamily: {
        sans: [
          'Inter',
          'SF Pro Display',
          'PingFang SC',
          '-apple-system',
          'sans-serif',
        ],
        mono: [
          'JetBrains Mono',
          'ui-monospace',
          'monospace',
        ],
      },
      letterSpacing: {
        tightest: '-0.038em',
        tighter: '-0.02em',
        tight: '-0.011em',
      }
    },
  },
  plugins: [],
}
