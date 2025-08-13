package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/flowhunt/boilerplate/linkbuilding"
)

type KeywordData struct {
	Keywords []linkbuilding.Keyword `json:"keywords"`
}

func main() {
	var (
		inputFile    = flag.String("input", "", "Input HTML file (or stdin if empty)")
		outputFile   = flag.String("output", "", "Output HTML file (or stdout if empty)")
		keywordsFile = flag.String("keywords", "", "JSON file with keywords")
		currentURL   = flag.String("url", "", "Current page URL")
		skipFlag     = flag.Bool("skip", false, "Skip linkbuilding")
	)
	flag.Parse()

	// Read input HTML
	var input []byte
	var err error
	
	if *inputFile == "" {
		input, err = io.ReadAll(os.Stdin)
	} else {
		input, err = os.ReadFile(*inputFile)
	}
	
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}

	// If skip flag is set, just output the input as-is
	if *skipFlag {
		if *outputFile == "" {
			fmt.Print(string(input))
		} else {
			err = os.WriteFile(*outputFile, input, 0644)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
				os.Exit(1)
			}
		}
		return
	}

	// Read keywords
	var keywords []linkbuilding.Keyword
	
	if *keywordsFile != "" {
		keywordData, err := os.ReadFile(*keywordsFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading keywords file: %v\n", err)
			os.Exit(1)
		}
		
		var kd KeywordData
		err = json.Unmarshal(keywordData, &kd)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing keywords JSON: %v\n", err)
			os.Exit(1)
		}
		keywords = kd.Keywords
	}

	// Process HTML
	config := linkbuilding.DefaultConfig()
	processed, err := linkbuilding.ProcessHTMLString(string(input), keywords, config, *currentURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error processing HTML: %v\n", err)
		os.Exit(1)
	}

	// Write output
	if *outputFile == "" {
		fmt.Print(processed)
	} else {
		err = os.WriteFile(*outputFile, []byte(processed), 0644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
			os.Exit(1)
		}
	}
}