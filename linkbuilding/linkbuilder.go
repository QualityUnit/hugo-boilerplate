package linkbuilding

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"golang.org/x/net/html"
)

// Keyword represents a keyword to be linked
type Keyword struct {
	Keyword  string
	URL      string
	Title    string
	Exact    bool
	Priority int
}

// Config holds the linkbuilding configuration
type Config struct {
	MaxSameKeywordReplacements        int
	MaxSameURLReplacements           int
	MaxKeywordURLReplacements        int
	MaxTotalReplacements            int
	CharactersPerLinkDensity        int
	MinParagraphLength              int
	SkipLinkbuilding               bool
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		MaxSameKeywordReplacements:  5,
		MaxSameURLReplacements:      3,
		MaxKeywordURLReplacements:   1,
		MaxTotalReplacements:       50,
		CharactersPerLinkDensity:   5,
		MinParagraphLength:         30,
		SkipLinkbuilding:          false,
	}
}

// LinkBuilder processes HTML content and adds links
type LinkBuilder struct {
	config           *Config
	keywords         []Keyword
	keywordCounts    map[string]int
	urlCounts        map[string]int
	keywordURLCounts map[string]int
	totalLinks       int
	currentPageURL   string
}

// New creates a new LinkBuilder
func New(config *Config, keywords []Keyword, currentPageURL string) *LinkBuilder {
	if config == nil {
		config = DefaultConfig()
	}
	
	// Sort keywords by priority (higher first) and length (longer first)
	sortedKeywords := make([]Keyword, len(keywords))
	copy(sortedKeywords, keywords)
	
	// Simple bubble sort for demonstration
	for i := 0; i < len(sortedKeywords); i++ {
		for j := i + 1; j < len(sortedKeywords); j++ {
			if sortedKeywords[j].Priority > sortedKeywords[i].Priority ||
				(sortedKeywords[j].Priority == sortedKeywords[i].Priority && 
				 len(sortedKeywords[j].Keyword) > len(sortedKeywords[i].Keyword)) {
				sortedKeywords[i], sortedKeywords[j] = sortedKeywords[j], sortedKeywords[i]
			}
		}
	}
	
	return &LinkBuilder{
		config:           config,
		keywords:         sortedKeywords,
		keywordCounts:    make(map[string]int),
		urlCounts:        make(map[string]int),
		keywordURLCounts: make(map[string]int),
		totalLinks:       0,
		currentPageURL:   currentPageURL,
	}
}

// ProcessHTML processes HTML content and adds links based on keywords
func (lb *LinkBuilder) ProcessHTML(htmlContent string) (string, error) {
	// If linkbuilding is disabled, return content as-is
	if lb.config.SkipLinkbuilding {
		return htmlContent, nil
	}

	// Parse HTML
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(htmlContent))
	if err != nil {
		return htmlContent, fmt.Errorf("failed to parse HTML: %w", err)
	}

	// Process text nodes
	lb.processNode(doc.Selection)

	// Get the processed HTML
	html, err := doc.Html()
	if err != nil {
		return htmlContent, fmt.Errorf("failed to serialize HTML: %w", err)
	}

	return html, nil
}

// processNode recursively processes DOM nodes
func (lb *LinkBuilder) processNode(s *goquery.Selection) {
	// Skip if we've reached the maximum total links
	if lb.totalLinks >= lb.config.MaxTotalReplacements {
		return
	}

	s.Contents().Each(func(i int, sel *goquery.Selection) {
		node := sel.Get(0)
		if node == nil {
			return
		}

		// Skip certain elements
		parent := sel.Parent()
		parentTag := goquery.NodeName(parent)
		
		// Don't process text inside links, headings, scripts, styles, etc.
		if parentTag == "a" || parentTag == "h1" || parentTag == "h2" || 
		   parentTag == "h3" || parentTag == "h4" || parentTag == "h5" || 
		   parentTag == "h6" || parentTag == "script" || parentTag == "style" ||
		   parentTag == "code" || parentTag == "pre" {
			return
		}

		if node.Type == html.TextNode {
			text := node.Data
			
			// Skip short text
			trimmedText := strings.TrimSpace(text)
			if len(trimmedText) < lb.config.MinParagraphLength {
				return
			}

			// Process keywords - use original text to preserve whitespace
			processedText := lb.processText(text)
			
			// If text was modified, replace the node
			if processedText != text {
				// Parse the new HTML fragment
				newNodes, err := html.ParseFragment(strings.NewReader(processedText), &html.Node{
					Type:     html.ElementNode,
					Data:     "body",
					DataAtom: 0,
				})
				
				if err == nil && len(newNodes) > 0 {
					// Create a new parent to hold the nodes
					wrapper := &html.Node{
						Type: html.ElementNode,
						Data: "span",
					}
					
					for _, n := range newNodes {
						wrapper.AppendChild(n)
					}
					
					// Replace the text node with the new nodes
					node.Parent.InsertBefore(wrapper, node)
					node.Parent.RemoveChild(node)
					
					// Unwrap the span to keep the DOM clean
					for child := wrapper.FirstChild; child != nil; {
						next := child.NextSibling
						node.Parent.InsertBefore(child, wrapper)
						child = next
					}
					node.Parent.RemoveChild(wrapper)
				}
			}
		} else if node.Type == html.ElementNode {
			// Recursively process child elements
			lb.processNode(sel)
		}
	})
}

// processText processes text content and adds links
func (lb *LinkBuilder) processText(text string) string {
	result := text
	
	for _, keyword := range lb.keywords {
		// Skip if this is the current page URL
		if keyword.URL == lb.currentPageURL {
			continue
		}
		
		// Check if we can add more links for this keyword/URL combination
		keywordCount := lb.keywordCounts[keyword.Keyword]
		urlCount := lb.urlCounts[keyword.URL]
		keywordURLKey := fmt.Sprintf("%s|%s", keyword.Keyword, keyword.URL)
		keywordURLCount := lb.keywordURLCounts[keywordURLKey]
		
		if keywordCount >= lb.config.MaxSameKeywordReplacements ||
		   urlCount >= lb.config.MaxSameURLReplacements ||
		   keywordURLCount >= lb.config.MaxKeywordURLReplacements ||
		   lb.totalLinks >= lb.config.MaxTotalReplacements {
			continue
		}
		
		// Create regex pattern
		var pattern *regexp.Regexp
		escapedKeyword := regexp.QuoteMeta(keyword.Keyword)
		
		if keyword.Exact {
			pattern = regexp.MustCompile(`\b` + escapedKeyword + `\b`)
		} else {
			pattern = regexp.MustCompile(`(?i)\b` + escapedKeyword + `\b`)
		}
		
		// Check if keyword exists in text
		if !pattern.MatchString(result) {
			continue
		}
		
		// Create the link HTML
		title := html.EscapeString(keyword.Title)
		if title == "" {
			title = keyword.Keyword
		}
		
		// Replace first occurrence only
		matches := pattern.FindStringIndex(result)
		if matches != nil {
			matchedText := result[matches[0]:matches[1]]
			replacement := fmt.Sprintf(`<a href="%s" title="%s" class="link-building-link">%s</a>`,
				keyword.URL, title, matchedText)
			
			result = result[:matches[0]] + replacement + result[matches[1]:]
			
			// Update counters
			lb.keywordCounts[keyword.Keyword]++
			lb.urlCounts[keyword.URL]++
			lb.keywordURLCounts[keywordURLKey]++
			lb.totalLinks++
		}
	}
	
	return result
}

// ProcessHTMLString is a convenience function for processing HTML strings
func ProcessHTMLString(htmlContent string, keywords []Keyword, config *Config, currentPageURL string) (string, error) {
	lb := New(config, keywords, currentPageURL)
	return lb.ProcessHTML(htmlContent)
}