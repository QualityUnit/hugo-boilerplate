/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './layouts/**/*.html',
    './content/**/*.{html,md}',
    '../../layouts/**/*.html',
    '../../content/**/*.{html,md}',
    '../**/layouts/**/*.html',
    '../**/content/**/*.{html,md}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#1a73e8',
          light: '#4285f4',
          dark: '#1557b0',
        },
        secondary: {
          DEFAULT: '#34a853',
        },
        accent: '#fbbc05',
        background: {
          DEFAULT: '#ffffff',
        },
        text: {
          DEFAULT: '#202124',
        },
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(to right, #1a73e8, #4285f4)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
    require('@tailwindcss/aspect-ratio'),
  ],
  // Ensure Tailwind doesn't conflict with the lazy loading and responsive image features
  safelist: ['lazy-load', 'lazy-load-bg', 'lazy-load-video', 'picture', 'source', 'webp', 'srcset'],
};
