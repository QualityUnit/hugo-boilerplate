# Linkbuilding Module for Hugo

This is a Go module that provides DOM-based linkbuilding functionality for Hugo sites. It parses HTML content, identifies keywords, and adds links automatically while respecting the HTML structure.

## Features

- DOM-based processing (no regex on raw HTML)
- Respects existing links and headings
- Configurable limits for link density
- Case-sensitive and case-insensitive matching
- Priority-based keyword processing

## Usage

### As a Go Library

```go
import "github.com/flowhunt/boilerplate/linkbuilding"

html := `<p>Learn about AI workflows and machine learning.</p>`
keywords := []linkbuilding.Keyword{
    {Keyword: "AI workflows", URL: "/ai", Title: "AI Guide"},
}
config := linkbuilding.DefaultConfig()

result, err := linkbuilding.ProcessHTMLString(html, keywords, config, "/current-page")
```

### As a Command Line Tool

```bash
# Build the tool
go build -o linkbuilder ./cmd/linkbuilder

# Process HTML file
./linkbuilder -input page.html -output processed.html -keywords keywords.json -url "/current-page"

# Process from stdin to stdout
echo "<p>AI workflows are great</p>" | ./linkbuilder -keywords keywords.json
```

### Keywords JSON Format

```json
{
  "keywords": [
    {
      "Keyword": "AI workflows",
      "URL": "/knowledge-base/ai-workflows/",
      "Title": "Learn about AI workflows",
      "Exact": false,
      "Priority": 1
    }
  ]
}
```

## Configuration

The module supports various configuration options:

- `MaxSameKeywordReplacements`: Maximum times the same keyword can be linked (default: 5)
- `MaxSameURLReplacements`: Maximum times the same URL can be linked (default: 3)
- `MaxKeywordURLReplacements`: Maximum times the same keyword-URL combo can be linked (default: 1)
- `MaxTotalReplacements`: Maximum total links per page (default: 50)
- `MinParagraphLength`: Minimum text length to process (default: 30)
- `SkipLinkbuilding`: Skip all processing when true (default: false)

## Integration with Hugo

This module can be integrated with Hugo in several ways:

1. **Post-processing**: Run the linkbuilder tool on generated HTML files after Hugo builds
2. **Custom shortcode**: Create a Hugo shortcode that calls the Go module
3. **Build pipeline**: Integrate into your CI/CD pipeline

### Example Post-processing Script

```bash
#!/bin/bash
# Run after hugo build

find public -name "*.html" -type f | while read file; do
    ./linkbuilder -input "$file" -output "$file.tmp" -keywords data/linkbuilding/en.json
    mv "$file.tmp" "$file"
done
```

## Testing

Run tests with:

```bash
go test -v
```

## License

Part of the Hugo Boilerplate theme.