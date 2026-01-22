# Hugo Boilerplate Theme

A clean and minimal Hugo theme designed for QualityUnit websites with a focus on performance, SEO, and responsive design. 
This theme includes Tailwind CSS integration, comprehensive SEO features, responsive image processing, and multilingual support out of the box.


## Projects running this theme
- [AiMingle](https://www.aimingle.cz/) - https://github.com/QualityUnit/aimingle-hugo
- [FlowHunt](https://www.flowhunt.io/) - https://github.com/QualityUnit/flowhunt-hugo
- [Post Affiliate Pro](https://main.d1jnujto7ausyo.amplifyapp.com/) - https://github.com/QualityUnit/postaffiliatepro-hugo
- [UrlsLab](https://www.urlslab.com/) - https://github.com/QualityUnit/urlslab-hugo
- [PhotomaticAI](https://www.photomaticai.com/) - https://github.com/QualityUnit/photomaticai-hugo/
- [Wachman](https://www.wachman.eu) - https://github.com/vzeman/wachman

## Notes For developers
- Many partials and shortcodes are not correct, we need to fix them.
- If shortcode or partial is used already in any project, always make sure your changes are compatible with old data, new parameters needs to be optional
- Make sure all texts can be translated, add texts to translation files in theme

## Features

- **Tailwind CSS Integration** - we have bought license from [https://tailwindcss.com/plus/ui-blocks/marketing](https://tailwindcss.com/plus/ui-blocks/marketing), we should try to keep the similar naming convention for our own elements
- **Responsive Design** - Optimized for all device sizes
- **Multilingual Support** - Built-in support for multiple languages
- **SEO Optimized** - Preprocessing of data done by `scripts/build_content.sh` script (linkbuilding, relations of content, image preprocessing, attributes syncing)
- **Responsive Images** - Automatic image processing with WebP conversion (`scripts/build_content.sh`)
- **Lazy Loading** - Performance-optimized image and video loading (shorcode `lazyimg` should be used in markdown and in partials)
- **Glossary System** - Built-in glossary with alphabetical navigation - this is just example post type, we can add more post types to share them accross projects
- **Tag & Category System** - Comprehensive taxonomy management, custom taxonomies are allowed per domain
- **Component Library** - Extensive collection of pre-built components:
  - Headers and navigation menus
  - Product showcases and listings
  - Feature sections
  - Review components
  - Banners and CTAs

## Content Preparation Scripts

The theme includes a comprehensive set of scripts in the `scripts/` directory that prepare your content for optimal performance and SEO.

### Main Script: build_content.sh

This is the primary script that coordinates the entire content preparation process:

```bash
# Run from the Hugo site root
./themes/boilerplate/scripts/build_content.sh
```

#### What build_content.sh Does:

1. **Sets up environment**: Creates a Python virtual environment and installs required dependencies
2. **Syncs translations**: Ensures translation keys are consistent across language files
3. **Builds Hugo site**: Validates content by building the entire site (exits on errors)
4. **Offloads images**: Downloads and stores images from external sources if needed
5. **Finds duplicate images**: Identifies duplicate images across the project
6. **Translates missing content**: Uses the FlowHunt API to translate missing content files
7. **Synchronizes attributes**: Ensures content attributes are consistent across translations
8. **Generates translation URLs**: Creates URL mapping for all languages
9. **Generates related content**: Creates YAML files for internal linking
10. **Extracts automatic links**: Extracts keywords from frontmatter for linkbuilding (first 2 keywords per page)
11. **Precomputes linkbuilding**: Optimizes keyword files based on actual content
    - **Incremental processing**: Only processes languages without existing optimized files
    - **Force recomputation**: Delete specific `*_optimized.json` files to recompute those languages
    - **Force all**: Use `--force` flag to recompute all languages
12. **Preprocesses images**: Optimizes images for web delivery (WebP conversion, responsive sizes)

#### Running Specific Steps:

You can run specific parts of the build process using the `--step` flag:

```bash
# Run only image preprocessing
./themes/boilerplate/scripts/build_content.sh --step preprocess_images

# Run multiple steps
./themes/boilerplate/scripts/build_content.sh --step sync_translations,validate_content
```

Available steps:
- `sync_translations`: Synchronize translation keys across files
- `build_hugo`: Build and validate Hugo site (exits on errors)
- `offload_images`: Download images from external services
- `find_duplicate_images`: Find and report duplicate images
- `translate`: Translate missing content with FlowHunt API
- `sync_content_attributes`: Ensure content attribute consistency
- `generate_translation_urls`: Generate URL mappings for all languages
- `generate_related_content`: Create related content data
- `extract_automatic_links`: Extract keywords from frontmatter for linkbuilding
- `precompute_linkbuilding`: Optimize linkbuilding files based on actual content
- `apply_linkbuilding`: Apply linkbuilding to HTML files in public folder
- `preprocess_images`: Optimize images for web delivery

#### Requirements:

- Python 3.8+ with pip
- FlowHunt API key (for translation functionality)
- Image processing tools (handled by the script)

The script will prompt for a FlowHunt API key if not already configured.

### Linkbuilding Optimization

The linkbuilding system includes smart optimization features:

#### Incremental Processing
The `precompute_linkbuilding` step only processes languages that don't have optimized files yet:
- Automatically skips languages with existing `*_optimized.json` files
- Significantly speeds up repeated builds
- Only recomputes when content changes

#### Force Recomputation
To recompute specific languages:
```bash
# Method 1: Delete the optimized file for that language
rm data/linkbuilding/optimized/de_optimized.json
./themes/boilerplate/scripts/build_content.sh --step precompute_linkbuilding

# Method 2: Use force flag for specific languages
themes/boilerplate/scripts/.venv/bin/python themes/boilerplate/scripts/precompute_linkbuilding.py \
  --linkbuilding-dir data/linkbuilding \
  --public-dir public \
  --output-dir data/linkbuilding/optimized \
  --force-languages de fr es

# Method 3: Force all languages
themes/boilerplate/scripts/.venv/bin/python themes/boilerplate/scripts/precompute_linkbuilding.py \
  --linkbuilding-dir data/linkbuilding \
  --public-dir public \
  --output-dir data/linkbuilding/optimized \
  --force
```

#### Performance Features
- Processes 3 languages in parallel by default (configurable with `--parallel-languages`)
- Skips unnecessary files (categories, tags, pagination)
- Early stopping when all keywords are found
- Deduplicates case-insensitive keywords automatically

## Installation

### Option 1: As a Git Submodule (Recommended)

This method allows you to easily update the theme when new versions are released:

```bash
# Navigate to your Hugo site's root directory
cd your-hugo-site

# Add the theme as a submodule
git submodule add https://github.com/qualityunit/hugo-boilerplate.git themes/boilerplate

# Update your Hugo configuration to use the theme
echo 'theme = "boilerplate"' >> hugo.toml
```

### Option 2: Manual Download

If you prefer not to use Git submodules:

```bash
# Navigate to your Hugo site's root directory
cd your-hugo-site

# Download the theme
mkdir -p themes
curl -L https://github.com/owner/hugo-boilerplate/archive/main.tar.gz | tar -xz -C themes
mv themes/hugo-boilerplate-main themes/boilerplate

# Update your Hugo configuration to use the theme
echo 'theme = "boilerplate"' >> hugo.toml
```

## Dependencies

This theme requires Node.js and npm for build process orchestration. The project uses Gulp for task management, Tailwind CSS for styling, and ESBuild for JavaScript bundling.

You need to create a `package.json` file in your project root (not in the theme directory) with the necessary dependencies.

### Required Dependencies

```json
{
  "name": "your-project-name",
  "version": "1.0.0",
  "description": "Your project description",
  "scripts": {
    "start": "gulp",
    "dev": "gulp dev",
    "watch": "gulp watch",
    "build": "gulp css && gulp js"
  },
  "devDependencies": {
    "@tailwindcss/aspect-ratio": "^0.4.2",
    "@tailwindcss/forms": "^0.5.7",
    "@tailwindcss/typography": "^0.5.10",
    "autoprefixer": "^10.4.21",
    "cssnano": "^7.0.7",
    "esbuild": "^0.25.5",
    "gulp": "^5.0.1",
    "gulp-postcss": "^10.0.0",
    "postcss": "^8.4.31",
    "postcss-cli": "^10.1.0",
    "postcss-import": "^16.1.0",
    "tailwindcss": "^3.4.17",
    "yargs": "^18.0.0"
  }
}
```

### Build System Overview

The theme uses a Gulp-based build system that:

1. **CSS Processing**: 
   - Processes CSS `@import` statements with postcss-import
   - Compiles Tailwind CSS with PostCSS
   - Adds vendor prefixes with autoprefixer
   - Minifies output with cssnano
2. **JavaScript Bundling**: Uses ESBuild for fast bundling and minification
3. **Watch Mode**: Provides live reloading during development
4. **Hugo Integration**: Automatically starts Hugo server with configurable options

### Gulpfile Configuration

The build system requires a `gulpfile.js` in your project root:

```javascript
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

// CSS and JS source paths
const cssSrc = 'themes/boilerplate/assets/css/main.css';
const cssDest = 'static/css';
const jsEntryPoints = ['themes/boilerplate/assets/js/main.js'];
const jsDest = 'static/js';

// CSS build with @import processing
function buildCSS() {
    return gulp.src(cssSrc)
        .pipe(postcss([
            postcssImport,    // Process @import statements first
            tailwindcss,
            autoprefixer,
            cssnano()
        ]))
        .pipe(gulp.dest(cssDest));
}

// Build tasks and configuration...
```

## PostCSS Configuration

### Required Setup

This theme uses Tailwind CSS which is processed through Gulp and PostCSS. You **must** create a `postcss.config.js` file in your project root (not in the theme directory) with the following content:

```javascript
// postcss.config.js in your project root
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

### Tailwind Configuration

The theme includes a pre-configured `tailwind.config.js` that:

1. Scans the correct template directories for class usage
2. Includes necessary Tailwind plugins
3. Provides custom color schemes and utilities

```javascript
// tailwind.config.js example
module.exports = {
    darkMode: 'class',
    content: [
        './layouts/**/*.html',
        './themes/boilerplate/layouts/**/*.html', 
        './themes/boilerplate/assets/js/**/*.js',
    ],
    theme: {
        extend: {
            // Custom theme extensions
        },
    },
    plugins: [
        require('@tailwindcss/typography'),
        require('@tailwindcss/forms'),
        require('@tailwindcss/aspect-ratio'),
    ],
};
```

### Build Process Integration

The Gulp build system automatically:

1. **Processes @import statements**: Uses postcss-import to inline imported CSS files
2. **Compiles Tailwind CSS**: Runs Tailwind CSS compilation with PostCSS
3. **Applies Autoprefixer**: Adds vendor prefixes for browser compatibility  
4. **Minifies Output**: Uses cssnano for production-ready CSS
5. **Watches Changes**: Rebuilds CSS when source files change

### CSS Import System

The theme supports CSS imports in the main CSS file:

```css
/* themes/boilerplate/assets/css/main.css */
@import "./fonts.css";
@import "./typewriter.css";
@import "./lazy-images.css";
@import "./lazy-videos.css";

@tailwind base;
@tailwind components;
@tailwind utilities;
```

The `postcss-import` plugin automatically inlines these imports during the build process, creating a single minified CSS file in `static/css/main.css`.

## JavaScript Bundling with ESBuild

The theme uses ESBuild for fast JavaScript bundling and processing:

### ESBuild Configuration

JavaScript files are processed with the following settings:

- **Entry Points**: `themes/boilerplate/assets/js/main.js`
- **Output**: `static/js/` directory
- **Format**: IIFE (Immediately Invoked Function Expression)
- **Bundling**: Enabled for dependency resolution
- **Minification**: Enabled for production builds
- **Watch Mode**: Available for development

### JavaScript Development

1. **Main Entry**: Place your JavaScript in `themes/boilerplate/assets/js/main.js`
2. **Modules**: Import ES6 modules normally - ESBuild handles bundling
3. **Build**: Run `gulp js` or `npm run build` to process JavaScript
4. **Watch**: Use `gulp dev` for automatic rebuilding during development

### JavaScript Build Process

```javascript
// gulpfile.js JavaScript task example
function buildJS() {
    return esbuild.build({
        entryPoints: ['themes/boilerplate/assets/js/main.js'],
        bundle: true,
        minify: true,
        format: 'iife',
        outdir: 'static/js'
    });
}
```

### Troubleshooting Build Issues

If you encounter build errors:

1. **Verify Dependencies**: Ensure all npm packages are installed correctly with `npm install`
2. **Check File Paths**: Confirm Tailwind content paths match your project structure
3. **Gulp Configuration**: Verify gulpfile.js is properly configured with all required plugins
4. **PostCSS Config**: Ensure postcss.config.js exists in project root
5. **CSS Import Errors**: Check that imported CSS files exist in the correct paths relative to main.css

**Common Issues:**

- **MIME type errors**: Usually resolved by properly configuring postcss-import plugin
- **Missing CSS files**: Ensure all @import paths are correct and files exist
- **Build failures**: Check that all dependencies are installed and paths are correct

## Configuration

### Basic Configuration

This theme supports two configuration approaches:

#### Option 1: Using a Single Configuration File (Traditional)

Add the following to your `hugo.toml` file:

```toml
baseURL = 'https://example.com/'
languageCode = 'en-us'
title = 'Your Site Title'
theme = 'boilerplate'
defaultContentLanguage = "en"
defaultContentLanguageInSubdir = true

# Output formats configuration
[outputs]
  home = ["HTML", "RSS", "SITEMAP"]

# Site parameters
[params]
  description = "Your site description"
  author = "Your Name"
  mainSections = ["blog"]
  dateFormat = "January 2, 2006"
```

#### Option 2: Using Split Configuration Files (Recommended)

For better organization, you can split your configuration into multiple files in a `config/_default/` directory:

1. Copy the example configuration structure from the theme:

```bash
mkdir -p config/_default
cp -r themes/boilerplate/config_example/_default/* config/_default/
```

2. Edit the individual configuration files to customize your site:
   - `hugo.toml` - Basic site configuration
   - `languages.toml` - Multilingual settings
   - `menus.toml` - Navigation menu structure
   - `params.toml` - Site parameters and features
   - `outputFormats.toml` - Output format configuration
   - `markup.toml` - Content rendering settings
   - `module.toml` - Hugo modules configuration

This modular approach makes your configuration more maintainable, especially for complex sites.

### Multilingual Setup

The theme supports multiple languages out of the box. Configure them in `languages.toml` (if using split configuration) or in your `hugo.toml` under the `[languages]` section:

```toml
[languages]
  [languages.en]
    languageName = "English"
    title = "English Site Title"
    weight = 1
    contentDir = "content/en"
    baseURL = "https://example.com"
    [languages.en.params]
      bcp47Lang = "en-us"
      description = "English site description"

  [languages.de]
    languageName = "Deutsch"
    title = "Deutsche Site Title"
    weight = 2
    contentDir = "content/de"
    baseURL = "https://example.de"
    [languages.de.params]
      bcp47Lang = "de"
      description = "Deutsche Seitenbeschreibung"
```

### Menu Configuration

Define your site's navigation in `menus.toml` (if using split configuration) or in your `hugo.toml` under language-specific menu sections:

```toml
[languages.en.menu]
  [[languages.en.menu.main]]
    identifier = "home"
    name = "Home"
    url = "/"
    weight = 1
  [[languages.en.menu.main]]
    identifier = "blog"
    name = "Blog"
    
    weight = 2
```

### Image Processing Configuration

For optimal image processing, add the following to your `params.toml` (if using split configuration) or to the `[params]` section of your `hugo.toml`:

```toml
[params.imaging]
  resampleFilter = "Lanczos"
  quality = 85
  anchor = "smart"
  bgColor = "#ffffff"
  webpQuality = 85
```

## Content Structure

### Creating Blog Posts

Create a new blog post with:

```bash
hugo new content/en/blog/my-post.md
```

Front matter example:

```yaml
+++
title = 'My Post Title'
date = 2025-04-03T07:43:16+02:00

description = "A comprehensive description for SEO purposes"
keywords = ["keyword1", "keyword2", "keyword3"]
image = "/images/blog/featured-image.jpg"
tags = ["tag1", "tag2"]
categories = ["category1"]
+++

Your post content here...
```

### Creating Glossary Terms

Create a new glossary term with:

```bash
hugo new content/en/glossary/term-name.md
```
or just create the file in the `content/en/glossary/` directory with extension *.md.


Front matter example:

```yaml
+++
title = 'Term Name'
date = 2025-04-03T07:43:16+02:00

url = "glossary/term-name"
description = "A comprehensive description of the term for SEO purposes"
keywords = ["keyword1", "keyword2", "keyword3"]
image = "/images/glossary/term-image.jpg"
term = "Term Name"
shortDescription = "A brief description of the term"
category = "T"
tags = ["tag1", "tag2"]
additionalImages = [
  "/images/glossary/additional-image1.jpg",
  "/images/glossary/additional-image2.jpg"
]

# CTA Section Configuration
showCTA = true
ctaHeading = "Related CTA Heading"
ctaDescription = "Call to action description text"
ctaPrimaryText = "Primary Button"
ctaPrimaryURL = "/related-url/"
ctaSecondaryText = "Secondary Button"
ctaSecondaryURL = "/another-url/"

[[faq]]
question = "Frequently asked question about the term?"
answer = "Comprehensive answer to the question that provides valuable information about the term."
+++

# What is Term Name?

Main content about the term goes here...
```

## Automatic Linkbuilding

The theme provides an automatic linkbuilding feature that scans your content and replaces specified keywords with links. This is configured via YAML files in the `data/linkbuilding/` directory, with a separate file for each language (e.g., `en.yaml`, `de.yaml`).

### Configuration File Structure

Each language-specific YAML file should contain a list of `keywords`. Each keyword entry defines the term to be linked, the target URL, and other options.

Here's an example from `data/linkbuilding/en.yaml`:

```yaml
keywords:
  - keyword: "mcp"
    url: "/services/mcp-server-development/"
    exact: false
    priority: 1
    title: "We can develop and host your own MCP server"
  - keyword: "mcp server"
    url: "/services/mcp-server-development/"
    exact: false
    priority: 1
    title: "We can develop and host your own MCP server"
  - keyword: "mcp servers"
    url: "/services/mcp-server-development/"
    exact: false
    priority: 1
    title: "We can develop and host your own MCP server"
```

### Keyword Entry Fields:

-   `keyword`: (String) The actual word or phrase in your content that you want to turn into a link. The matching is case-insensitive by default.
-   `url`: (String) The destination URL for the link. This should typically be a site-relative path (e.g., `/services/your-service/`).
-   `exact`: (Boolean, optional, defaults to `false`) 
    -   If `false` (default): The keyword will be matched even if it's part of a larger word (e.g., if keyword is "log", "logging" would also be matched). The matching is case-insensitive.
    -   If `true`: The keyword will only be matched if it appears as an exact word (bounded by spaces or punctuation). The matching is case-sensitive.
-   `priority`: (Integer, optional, defaults to `1`) Used to determine which rule applies if multiple keywords could match the same text. Higher numbers usually mean higher priority. The linkbuilding module processes keywords based on their priority, with higher priority keywords being applied first.
-   `title`: (String, optional, defaults to the `keyword` value) The text to be used for the `title` attribute of the generated `<a>` HTML tag. This is often used for tooltips or to provide more context to search engines.

To add new linkbuilding rules, simply edit the appropriate `data/linkbuilding/<lang>.yaml` file and add new entries to the `keywords` list following this structure.

## Using Theme Components

### Shortcodes

The theme includes various shortcodes for common components:

```markdown
{{< products-with-image-grid
  background="bg-gray-50"
  product="{ title: 'Product Title', ... }" >}}

{{< features-with-split-image
  background="bg-white"
  heading="Feature Section Heading"
  description="Feature section description text" >}}
```

### Partials

You can include partials in your templates:

```go
{{ partial "layout/headers/centered_with_eyebrow.html" (dict
  "eyebrow" "Eyebrow Text"
  "heading" "Main Heading"
  "description" "Description text") }}
```

## Schema.org Structured Data (SEO)

The theme includes a comprehensive schema.org structured data implementation to improve SEO and search engine visibility. All schemas are automatically integrated and work out-of-the-box.

### What's Included

The theme provides **20+ schema.org types** specifically optimized for SaaS businesses:

#### Core Schemas (Automatic)
- **WebSite** - Site identity with search box functionality
- **Organization** - Company identity and contact information
- **WebPage** - Auto-detecting page types (AboutPage, ContactPage, FAQPage, etc.)
- **SoftwareApplication** - Product/software description

#### Content Schemas (Auto-detected by section)
- **Article/BlogPosting** - Blog posts and news articles
- **Course** - Academy and educational content
- **Service** - SaaS service offerings
- **Event** - Webinars and online events
- **JobPosting** - Career and job listings

#### Enhancement Schemas (Triggered by frontmatter)
- **Product** - Product listings and pricing
- **Offer** - Special promotions and deals
- **ItemList** - Feature lists, integration directories
- **HowTo** - Step-by-step guides
- **VideoObject** - Video content
- **FAQPage** - FAQ sections
- **QAPage** - Community Q&A
- **Review/AggregateRating** - Customer testimonials and ratings
- **Breadcrumbs** - Navigation breadcrumbs
- **Author/Person** - Content authors

### How It Works

Schema.org structured data is **automatically integrated** via the `schemas.html` partial which is included in the theme's `head.html`.

**Core schemas** (WebSite, Organization, WebPage) are always included on every page.

**Content-specific schemas** are triggered by **frontmatter parameters** - not by hardcoded section names. This makes the system work with any Hugo project structure.

Add parameters to your page frontmatter to enable specific schemas:
```yaml
---
title: "My Page"
schemaType: "Article"       # Triggers Article schema
author: "authorkey"         # Triggers Author schema
faq: [...]                  # Triggers FAQ schema
product: {...}              # Triggers Product schema
---
```

### Configuration

#### Site-Level Configuration

Add to your `config/_default/params.toml` or `hugo.toml`:

```toml
[params]
  # Enable site search box in Google
  searchUrl = "/search/"

  [params.organization]
    name = "Your Company Name"
    legalName = "Your Company Legal Name Inc."
    description = "Company description"
    logo = "/images/logo.png"
    email = "contact@example.com"
    telephone = "+1-555-555-5555"
    foundingDate = "2010-01-01"

    [params.organization.address]
      streetAddress = "123 Main St"
      addressLocality = "City"
      addressRegion = "State"
      postalCode = "12345"
      addressCountry = "US"

    [[params.organization.sameAs]]
      "https://facebook.com/yourcompany"
    [[params.organization.sameAs]]
      "https://twitter.com/yourcompany"
    [[params.organization.sameAs]]
      "https://linkedin.com/company/yourcompany"

  [params.software]
    name = "Your Software Name"
    description = "Your software description"
    applicationCategory = "BusinessApplication"
    operatingSystem = "Windows, macOS, Linux, Web"
    softwareVersion = "1.0.0"
    screenshot = "/images/screenshot.png"

    [params.software.offers]
      price = "99.00"
      priceCurrency = "USD"

    [params.software.aggregateRating]
      ratingValue = "4.8"
      reviewCount = "250"
      bestRating = "5"
```

#### Page-Level Configuration

Add schema data to your page frontmatter. Use the `schemaType` parameter to specify the main content type:

**Blog Posts/Articles:**
```yaml
---
title: "Article Title"
description: "Article description"
schemaType: "Article"          # or "BlogPosting", "TechArticle", "NewsArticle"
date: 2026-01-20
author: "authorkey"            # References data/authors.yaml
image: "/images/article.jpg"
tags: ["tag1", "tag2"]
---
```

**Course/Academy Pages:**
```yaml
---
title: "Course Title"
courseMode: "online"
educationalLevel: "Beginner"
timeRequired: "PT4H"
---
```

**Event/Webinar Pages:**
```yaml
---
title: "Webinar Title"
event:
  name: "SaaS Marketing Webinar"
  description: "Learn how to grow your SaaS"
  startDate: "2026-02-15T14:00:00-05:00"
  endDate: "2026-02-15T15:00:00-05:00"
  eventAttendanceMode: "OnlineEventAttendanceMode"
  location: "https://yoursite.com/webinar"
  offers:
    price: "0"
    priceCurrency: "USD"
---
```

**FAQ Sections:**
```yaml
---
title: "FAQ"
faq:
  - question: "How does it work?"
    answer: "It works by..."
  - question: "What is the price?"
    answer: "The price is..."
---
```

**Product Pages:**
```yaml
---
title: "Product Name"
product:
  name: "Product Name"
  description: "Product description"
  sku: "PROD-001"
  offers:
    price: "99.00"
    priceCurrency: "USD"
---
```

**Special Offers:**
```yaml
---
title: "Special Offer"
offer:
  name: "50% Off Annual Plan"
  price: "499"
  priceCurrency: "USD"
  priceValidUntil: "2026-12-31"
---
```

### Using Schemas in Custom Layouts

While schemas are automatically included via `head.html`, you can manually include specific schemas in custom layout files:

**Include a specific schema:**
```go
{{/* Single service */}}
{{ partial "schemaorg/service.html" (dict "service" .Params.service) }}

{{/* Multiple products */}}
{{ partial "schemaorg/product.html" (dict "products" .Params.products) }}

{{/* Event with custom data */}}
{{ partial "schemaorg/event.html" (dict
  "name" "Event Name"
  "description" "Event description"
  "startDate" "2026-02-15T14:00:00-05:00"
) }}
```

**Available schema partials:**
All schema partials are located in `themes/boilerplate/layouts/partials/schemaorg/`:
- `article.html` - Blog posts and articles
- `aggregaterating.html` - Ratings and reviews
- `author.html` - Content authors
- `breadcrumbs.html` - Navigation breadcrumbs
- `course.html` - Educational content
- `discussion.html` - Forum discussions
- `event.html` - Events and webinars
- `faq.html` - FAQ pages
- `howto.html` - Step-by-step guides
- `itemlist.html` - Lists of items
- `jobposting.html` - Job listings
- `offer.html` - Special offers
- `organization.html` - Company information
- `product.html` - Products
- `qapage.html` - Q&A pages
- `reviews.html` - Customer reviews
- `service.html` - Service offerings
- `softwareapplication.html` - Software products
- `video.html` - Video content
- `webpage.html` - Web pages
- `website.html` - Website identity

### Frontmatter Parameters Reference

The system uses frontmatter parameters to trigger schemas - no hardcoded section names:

| Parameter | Schema Type | Example |
|-----------|-------------|---------|
| `schemaType: "Article"` | Article/BlogPosting | Blog posts, news |
| `schemaType: "TechArticle"` | TechArticle | Technical guides |
| `author: "authorkey"` | Person/Author | Any content with author |
| `faq: [...]` | FAQPage | FAQ sections |
| `product: {...}` | Product | Product pages |
| `service: {...}` | Service | Service offerings |
| `event: {...}` | Event | Events, webinars |
| `courseMode: "online"` | Course | Educational content |
| `video: {...}` | VideoObject | Video content |
| `howto: {...}` | HowTo | Step-by-step guides |
| `jobPosting: {...}` | JobPosting | Job listings |
| `offer: {...}` | Offer | Special promotions |
| `testimonials: [...]` | Review | Customer reviews |
| `itemList: [...]` | ItemList | Lists of items |

**Note:** Individual page templates can also explicitly include schemas when needed.

### Testing Your Schemas

Validate your structured data using:

1. **Google Rich Results Test**: https://search.google.com/test/rich-results
2. **Schema.org Validator**: https://validator.schema.org/
3. **Google Search Console**: Monitor rich results performance

### Complete Documentation

For detailed documentation including all available parameters, examples, and configuration options, see:
`themes/boilerplate/layouts/partials/schemaorg/README.md`

### Benefits

Proper schema.org implementation provides:
- **Better SEO**: Improved search engine understanding of your content
- **Rich Results**: Enhanced search results with ratings, prices, events, etc.
- **Knowledge Graph**: Potential inclusion in Google Knowledge Graph
- **Voice Search**: Better optimization for voice assistants
- **AI Discoverability**: Improved visibility in AI-driven search (ChatGPT, Perplexity, etc.)

## Customization

### Tailwind Configuration

The theme uses Tailwind CSS. You can customize the Tailwind configuration by editing the `tailwind.config.js` file in the theme directory.

### CSS Customization

Add custom CSS by creating a file at `assets/css/custom.css` in your project root.

### Layout Customization

Override any theme layout by creating a matching file structure in your project's `layouts` directory.

## Troubleshooting

### HUGO Speed

To start server with debug log:

```bash
hugo server --gc --templateMetrics --templateMetricsHints  --logLevel debug
```

### Future date
Dont forget future date is not built by default, if you want to build future posts, you need to add `--buildFuture` flag:

```bash
hugo server --buildFuture
```


### Printing DEBUG messages during development
To print debug messages during development, you can use the `{{ printf }}` function in your templates:

```go
{{ warnf "DEBUG get-language-url: jsonify langData: %s" (jsonify $langData) }}
```


## Project Initialization

### Quick Start

```bash
# 1. Checkout from git
git clone your-repo-url
cd your-project

# 2. Initialize git submodules
git submodule update --init --recursive

# 3. Install dependencies
npm install

# 4. Build assets
npm run build

# 5. Start development server
npm run dev
```

### Development Workflow

#### Development Mode (with live reload)
```bash
# Start development server with watch mode (all languages)
npm run dev

# Or with specific options
gulp dev --en           # English only (faster)
gulp dev --metrics      # Show template metrics
```

#### Production Build
```bash
# Build assets for production
npm run build

# Start server without watch
npm start
```

#### Asset-only Operations
```bash
# Build CSS only
gulp css

# Build JavaScript only
gulp js

# Watch for changes without server
npm run watch
```

### Gulp Tasks Available

- `gulp` (default) - Build assets and start Hugo server
- `gulp dev` - Development mode with watch and live reload
- `gulp css` - Build CSS with Tailwind processing
- `gulp js` - Bundle JavaScript with ESBuild
- `gulp watch` - Watch mode for asset changes only

### Server Options

The Gulp configuration supports various Hugo server options:

- `--en` - English only (faster startup, less memory usage)
- `--metrics` - Show template metrics and hints
- Default - All languages (complete multilingual experience)


## Generating new content using FlowHunt Flows
1. copy csv file with columns: flow_input, filename to e.g. scripts directory in theme
2. go to scripts directory `cd themes/boilerplate/scripts`
3. activate python environment:
```bash
source .venv/bin/activate
```
3. run the script to generate content:
```bash
python generate_content.py --input_file mcp_servers_smart_filtered.csv --flow_id 53849f45-6749-42cd-a27c-29562a25998f --output_dir ../../../content/en/mcp-servers 
```
IMPORTANT: you need in .env file a FlowHunt API key set as `FLOWHUNT_API_KEY=your_api_key_here`, API key needs to be from same workspace as the flow you are using to generate content.


### Common Issues

1. **PostCSS Processing Errors**: Ensure you have the correct PostCSS configuration in your project root.

2. **Image Processing Issues**: Check that Hugo has the required permissions to process images.

3. **Multilingual Configuration**: Verify that your content directories match the configured language directories.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENSE) for details.
