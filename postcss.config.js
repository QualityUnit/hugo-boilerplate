/*
  Tailwind v4: `@tailwindcss/postcss` replaces the `tailwindcss` plugin and has
  `@import` inlining plus vendor prefixing (via Lightning CSS) built in, so
  `postcss-import` and `autoprefixer` are no longer needed.

  The Tailwind config is named from CSS via `@config` in assets/css/main.css, not
  passed here. Consuming projects generally build through their own gulp/PostCSS
  setup and never read this file.
*/
module.exports = {
  plugins: {
    '@tailwindcss/postcss': {},
  },
}
