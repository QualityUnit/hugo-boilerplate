/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class', // important so that dark: utilities work on the .dark class

  content: [
    './layouts/**/*.html',
    '../../layouts/**/*.html',
    '../**/layouts/**/*.html',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        primary: {
            50: 'rgb(var(--color-primary-50) / <alpha-value>)',
            100: 'rgb(var(--color-primary-100) / <alpha-value>)',
            200: 'rgb(var(--color-primary-200) / <alpha-value>)',
            300: 'rgb(var(--color-primary-300) / <alpha-value>)',
            400: 'rgb(var(--color-primary-400) / <alpha-value>)',
            500: 'rgb(var(--color-primary-500) / <alpha-value>)',
            600: 'rgb(var(--color-primary-600) / <alpha-value>)',
            700: 'rgb(var(--color-primary-700) / <alpha-value>)',
            800: 'rgb(var(--color-primary-800) / <alpha-value>)',
            900: 'rgb(var(--color-primary-900) / <alpha-value>)',
            950: 'rgb(var(--color-primary-950) / <alpha-value>)',
            DEFAULT: 'rgb(var(--color-primary) / <alpha-value>)', // (600) as default
        },
        // 🎨 COMPLETE BUTTON COLOR SYSTEM - All button types
        'button-primary': {
            DEFAULT: 'rgb(var(--color-button-primary) / <alpha-value>)',
            hover: 'rgb(var(--color-button-primary-hover) / <alpha-value>)',
            dark: 'rgb(var(--color-button-primary-dark) / <alpha-value>)',
            'dark-hover': 'rgb(var(--color-button-primary-dark-hover) / <alpha-value>)',
            outline: 'rgb(var(--color-button-primary-outline) / <alpha-value>)',
        },
        'button-secondary': {
            DEFAULT: 'rgb(var(--color-button-secondary-bg) / <alpha-value>)',
            text: 'rgb(var(--color-button-secondary-text) / <alpha-value>)',
            border: 'rgb(var(--color-button-secondary-border) / <alpha-value>)',
            'border-hover': 'rgb(var(--color-button-secondary-border-hover) / <alpha-value>)',
            'bg-hover': 'rgb(var(--color-button-secondary-bg-hover) / <alpha-value>)',
        },
        'button-secondary-dark': {
            DEFAULT: 'rgb(var(--color-button-secondary-dark-bg) / <alpha-value>)',
            text: 'rgb(var(--color-button-secondary-dark-text) / <alpha-value>)',
            border: 'rgb(var(--color-button-secondary-dark-border) / <alpha-value>)',
            'border-hover': 'rgb(var(--color-button-secondary-dark-border-hover) / <alpha-value>)',
            'bg-hover': 'rgb(var(--color-button-secondary-dark-bg-hover) / <alpha-value>)',
        },
        'button-text': {
            DEFAULT: 'rgb(var(--color-button-text) / <alpha-value>)',
            hover: 'rgb(var(--color-button-text-hover) / <alpha-value>)',
            dark: 'rgb(var(--color-button-text-dark) / <alpha-value>)',
            'dark-hover': 'rgb(var(--color-button-text-dark-hover) / <alpha-value>)',
        },
        // BRAND COLORS
        'brand': {
            surface: 'rgb(var(--color-brand-surface) / <alpha-value>)',
            'dark-surface': 'rgb(var(--color-brand-dark-surface) / <alpha-value>)',
            'surface-secondary': 'rgb(var(--color-brand-surface-secondary) / 20)',
            'dark-surface-secondary': 'rgb(var(--color-brand-dark-surface-secondary) / <alpha-value>)',
            border: 'rgb(var(--color-brand-border) / <alpha-value>)',
            'dark-border': 'rgb(var(--color-brand-dark-border) / <alpha-value>)',
            text: 'rgb(var(--color-brand-text) / <alpha-value>)',
            'dark-text': 'rgb(var(--color-brand-dark-text) / <alpha-value>)',
            icon: 'rgb(var(--color-brand-icon) / <alpha-value>)',
            'dark-icon': 'rgb(var(--color-brand-dark-icon) / <alpha-value>)',
            underlight: 'rgb(var(--color-brand-underlight) / <alpha-value>)',
            'dark-underlight': 'rgb(var(--color-brand-dark-underlight) / <alpha-value>)',
        },
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
