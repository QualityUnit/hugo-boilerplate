# Configuration Structure

This directory contains example configuration files for the Hugo Boilerplate theme.

## Usage

Instead of using a single `hugo.toml` file, this approach splits your configuration into multiple files for better organization and maintainability. This is especially helpful for complex sites with many settings.

### How to Use These Files

1. Copy this entire directory structure to your project root:

   ```bash
   mkdir -p config/_default
   cp -r themes/boilerplate/config_example/_default/* config/_default/
   ```

2. Edit each file to customize your site settings:

   - **hugo.toml** - Basic site configuration (baseURL, title, theme, etc.)
   - **languages.toml** - Multilingual settings
   - **menus.toml** - Navigation menu structure
   - **params.toml** - Site parameters and features
   - **outputFormats.toml** - Output format configuration
   - **markup.toml** - Content rendering settings
   - **module.toml** - Hugo modules configuration

### Advantages of This Approach

- **Cleaner Organization** - Logical separation of configuration by purpose
- **Easier Maintenance** - Shorter files that are focused on specific settings
- **Better Collaboration** - Reduced merge conflicts when multiple people work on the configuration
- **Improved Readability** - Easier to find specific settings when they're organized by category

For more information, see the theme's main README.md file or the [Hugo documentation on configuration](https://gohugo.io/getting-started/configuration/).
