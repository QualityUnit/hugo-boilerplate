const gulp = require('gulp');
const postcss = require('gulp-postcss');
const postcssImport = require('postcss-import');
const tailwindcss = require('tailwindcss');
const autoprefixer = require('autoprefixer');
const esbuild = require('esbuild');
const { spawn } = require('child_process');
const cssnano = require('cssnano');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

const argv = yargs(hideBin(process.argv)).argv;

const cssSrc = 'themes/boilerplate/assets/css/main.css';
const cssDest = 'static/css';
const jsEntryPoints = ['themes/boilerplate/assets/js/main.js'];
const jsDest = 'static/js';

function buildCSS() {
  return gulp.src(cssSrc)
    .pipe(postcss([
      postcssImport,
      tailwindcss,
      autoprefixer,
      cssnano()
    ]))
    .pipe(gulp.dest(cssDest));
}

function buildJS() {
  return esbuild.build({
    entryPoints: jsEntryPoints,
    bundle: true,
    minify: true,
    format: 'iife',
    outdir: jsDest
  });
}

function startHugoServer() {
  let args = ['server', '--buildDrafts', '--buildFuture', '--disableFastRender'];
  
  if (argv.en) {
    args.push('--config', 'hugo.toml', '--contentDir', 'content/en');
  }
  
  if (argv.metrics) {
    args.push('--templateMetrics', '--templateMetricsHints');
  }
  
  const hugo = spawn('hugo', args, { stdio: 'inherit' });
  
  hugo.on('close', (code) => {
    console.log(`Hugo process exited with code ${code}`);
  });
}

function watchFiles() {
  gulp.watch('themes/boilerplate/assets/css/**/*.css', buildCSS);
  gulp.watch('themes/boilerplate/assets/js/**/*.js', buildJS);
}

const build = gulp.series(buildCSS, buildJS);
const dev = gulp.series(build, gulp.parallel(watchFiles, startHugoServer));
const watch = gulp.series(build, watchFiles);

exports.css = buildCSS;
exports.js = buildJS;
exports.build = build;
exports.dev = dev;
exports.watch = watch;
exports.default = gulp.series(build, startHugoServer);
