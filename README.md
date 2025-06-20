# Hugo Boilerplate Theme

A clean and minimal Hugo theme designed for QualityUnit websites with a focus on performance, SEO, and responsive design. 
This theme includes Tailwind CSS integration, comprehensive SEO features, responsive image processing, and multilingual support out of the box.

## Projects running this theme
- [AiMingle](https://www.aimingle.cz/) - https://github.com/QualityUnit/aimingle-hugo
- [FlowHunt](https://www.flowhunt.io/) - https://github.com/QualityUnit/flowhunt-hugo
- [Post Affiliate Pro](https://www.postaffiliatepro.com/) - https://github.com/QualityUnit/postaffiliatepro-hugo
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
3. **Validates content**: Checks content structure and formatting
4. **Offloads images**: Downloads and stores images from external sources if needed
5. **Translates missing content**: Uses the FlowHunt API to translate missing content files
6. **Synchronizes attributes**: Ensures content attributes are consistent across translations
7. **Re-validates content**: Checks content structure again after translation
8. **Generates related content**: Creates YAML files for internal linking
9. **Preprocesses images**: Optimizes images for web delivery (WebP conversion, responsive sizes)

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
- `validate_content`: Validate content files before processing
- `offload_images`: Download images from external services
- `translate`: Translate missing content with FlowHunt API
- `sync_content_attributes`: Ensure content attribute consistency
- `validate_content_post`: Validate content after translation
- `generate_related_content`: Create related content data
- `preprocess_images`: Optimize images for web delivery

#### Requirements:

- Python 3.8+ with pip
- FlowHunt API key (for translation functionality)
- Image processing tools (handled by the script)

The script will prompt for a FlowHunt API key if not already configured.

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

This theme requires Node.js and npm for Tailwind CSS processing. You need to create a `package.json` file in your project root (not in the theme directory) with the necessary dependencies.

You can use the example provided in the theme at `themes/boilerplate/package.json.example`:

```bash
# Copy the example package.json to your project root
cp themes/boilerplate/package.json.example package.json

# Install dependencies
npm install
```

The minimum required dependencies are:

```json
{
  "devDependencies": {
    "@tailwindcss/aspect-ratio": "^0.4.2",
    "@tailwindcss/forms": "^0.5.7",
    "@tailwindcss/typography": "^0.5.10",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.31",
    "postcss-cli": "^10.1.0",
    "tailwindcss": "^3.3.5"
  }
}
```

## PostCSS Configuration

### Required Setup

This theme uses Tailwind CSS which requires PostCSS for processing. You **must** create a `postcss.config.js` file in your project root (not in the theme directory) with the following content:

```javascript
// postcss.config.js in your project root
module.exports = {
  plugins: {
    tailwindcss: {
      config: './themes/boilerplate/tailwind.config.js',
    },
    autoprefixer: {},
  },
};
```

This configuration tells PostCSS to:

1. Use the Tailwind CSS plugin with the configuration file located in the theme directory
2. Apply autoprefixer for cross-browser compatibility

### Troubleshooting PostCSS Issues

If you encounter errors related to PostCSS processing (such as `Cannot read properties of undefined (reading 'blocklist')`), check the following:

1. Ensure your `postcss.config.js` is in the project root, not just in the theme directory
2. Verify that all dependencies are installed correctly with `npm install`
3. Make sure the path to the Tailwind config file is correct relative to your project structure
4. Check for any path resolution issues in your project's JavaScript configuration files

### Advanced PostCSS Configuration

For advanced users who want to customize the PostCSS processing:

```javascript
// postcss.config.js with additional options
module.exports = {
  plugins: {
    'postcss-import': {},
    'tailwindcss/nesting': {},
    tailwindcss: {
      config: './themes/boilerplate/tailwind.config.js',
    },
    autoprefixer: {
      flexbox: 'no-2009',
    },
    'postcss-preset-env': {
      features: { 'nesting-rules': false },
    },
  },
};
```

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
draft = false
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
draft = false
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
-   `priority`: (Integer, optional, defaults to `1`) Used to determine which rule applies if multiple keywords could match the same text. Higher numbers usually mean higher priority, but the exact logic depends on the implementation in the `linkbuilding.html` partial. Currently, the partial processes keywords in the order they appear in the YAML file, and the first match for a given piece of text is used. The `priority` field itself is not directly used by the current version of the `linkbuilding.html` partial to reorder or select rules beyond their sequence in the file.
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


## Init project:
- checkout from git
- init git submodules `git submodule update --init --recursive`
- install dependencies `npm install`
- build css `npm run build:css`
- start server `hugo server --gc`


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
