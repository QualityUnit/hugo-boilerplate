/**
 * Typewriter Effect Component
 * Creates a typewriter animation effect for the last N words in a heading,
 * with an option to cycle through alternative words/phrases.
 */
class TypewriterEffect {
  constructor(element, options = {}) {
    this.element = element;
    this.options = {
      words: options.words || 1, // Number of words from original text to animate initially
      typingSpeed: options.typingSpeed || 100,
      deletingSpeed: options.deletingSpeed || 50,
      pauseDuration: options.pauseDuration || 2000,
      color: options.color || 'text-indigo-600',
      wordAlternatives: options.wordAlternatives || [], // Array of alternative strings
      ...options
    };
    
    this.originalText = this.element.textContent.trim();
    this.staticText = ''; // Part of originalText that remains static
    this.initialAnimatedTextPart = ''; // The first text to animate, derived from originalText

    this.animationSequence = []; // Holds all texts to be animated: [initialAnimatedTextPart, ...wordAlternatives]
    this.currentAnimationIndex = 0; // Index for animationSequence
    this.currentTextInSequence = ''; // The current string from animationSequence being animated

    this.currentCharIndex = 0;
    this.isDeleting = false;
    this.typewriterContainer = null;
    this.typewriterText = null;
    this.cursor = null;
    
    this.init();
  }
  
  init() {
    this.parseOriginalText();
    this.buildAnimationSequence();

    if (this.animationSequence.length === 0) {
      // If no text to animate (neither from original nor alternatives), restore original content and stop.
      this.element.textContent = this.originalText;
      return;
    }

    this.currentTextInSequence = this.animationSequence[this.currentAnimationIndex];

    this.createTypewriterElements();
    this.startAnimation();
  }
  
  parseOriginalText() {
    const words = this.originalText.split(' ').filter(w => w.length > 0);
    const wordsToAnimateCount = Math.min(this.options.words, words.length);
    
    if (wordsToAnimateCount > 0) {
      const wordsToAnimateArray = words.slice(-wordsToAnimateCount);
      this.initialAnimatedTextPart = wordsToAnimateArray.join(' ');
      this.staticText = words.slice(0, words.length - wordsToAnimateCount).join(' ');
    } else {
      this.initialAnimatedTextPart = '';
      this.staticText = this.originalText;
    }

    // Add space after static text if it exists and if there's something to animate next to it
    if (this.staticText && (this.initialAnimatedTextPart || (this.options.wordAlternatives && this.options.wordAlternatives.length > 0))) {
      if (this.staticText.length > 0 && !this.staticText.endsWith(' ')) {
        this.staticText += ' ';
      }
    }
  }

  buildAnimationSequence() {
    this.animationSequence = [];
    if (this.initialAnimatedTextPart) {
      this.animationSequence.push(this.initialAnimatedTextPart);
    }

    if (this.options.wordAlternatives && this.options.wordAlternatives.length > 0) {
      this.animationSequence.push(...this.options.wordAlternatives);
    }
  }

  createTypewriterElements() {
    // Create container for typewriter effect
    this.typewriterContainer = document.createElement('span');
    this.typewriterContainer.className = 'typewriter-container';
    
    // Create text element
    this.typewriterText = document.createElement('span');
    this.typewriterText.className = `typewriter-text ${this.options.color}`;
    
    // Create cursor element
    this.cursor = document.createElement('span');
    this.cursor.className = `typewriter-cursor ${this.options.color}`;
    this.cursor.textContent = '_';
    
    // Append elements
    this.typewriterContainer.appendChild(this.typewriterText);
    this.typewriterContainer.appendChild(this.cursor);
    
    // Replace original element content
    this.element.innerHTML = '';
    if (this.staticText) {
      this.element.appendChild(document.createTextNode(this.staticText));
    }
    this.element.appendChild(this.typewriterContainer);
    
    // Pre-render all possible words to calculate max width
    this.calculateMaxWidth();
  }
  
  calculateMaxWidth() {
    let maxWidth = 0;
    if (this.animationSequence.length > 0) {
        this.animationSequence.forEach(text => {
          this.typewriterText.textContent = text;
          const rect = this.typewriterText.getBoundingClientRect();
          if (rect.width > maxWidth) {
            maxWidth = rect.width;
          }
        });
    }

    this.typewriterContainer.style.minWidth = Math.ceil(maxWidth + 20) + 'px';
    this.typewriterText.textContent = '';
  }
  
  startAnimation() {
    if (!this.currentTextInSequence) return;

    this.animateText();
  }
  
  animateText() {
    if (this.isDeleting) {
      // Deleting animation
      this.typewriterText.textContent = this.currentTextInSequence.substring(0, this.currentCharIndex);
      this.currentCharIndex--;
      
      if (this.currentCharIndex < 0) {
        this.isDeleting = false;
        this.currentCharIndex = 0;
        
        this.currentAnimationIndex++;
        if (this.currentAnimationIndex >= this.animationSequence.length) {
          this.currentAnimationIndex = 0;
        }
        this.currentTextInSequence = this.animationSequence[this.currentAnimationIndex];

        if (!this.currentTextInSequence) { // Should not happen if sequence is built correctly
            this.cursor.classList.remove('typing');
            return;
        }

        setTimeout(() => this.animateText(), this.options.pauseDuration / 2);
        return;
      }
      
      setTimeout(() => this.animateText(), this.options.deletingSpeed);
    } else {
      // Typing animation
      this.cursor.classList.add('typing');
      this.typewriterText.textContent = this.currentTextInSequence.substring(0, this.currentCharIndex + 1);
      this.currentCharIndex++;
      
      if (this.currentCharIndex >= this.currentTextInSequence.length) {
        this.cursor.classList.remove('typing');
        this.isDeleting = true;
        
        setTimeout(() => this.animateText(), this.options.pauseDuration);
        return;
      }
      
      setTimeout(() => this.animateText(), this.options.typingSpeed);
    }
  }
  
  destroy() {
    // Restore original text
    this.element.textContent = this.originalText;
  }
}

// Auto-initialize typewriter effects
document.addEventListener('DOMContentLoaded', function() {
  const typewriterElements = document.querySelectorAll('[data-typewriter]');
  
  typewriterElements.forEach(element => {
    const alternativesString = element.dataset.typewriterWordAlternatives || '';
    const wordAlternatives = alternativesString
      ? alternativesString.split(',').map(s => s.trim()).filter(s => s.length > 0)
      : [];

    const options = {
      words: parseInt(element.dataset.typewriterWords) || 1,
      typingSpeed: parseInt(element.dataset.typewriterSpeed) || 100,
      deletingSpeed: parseInt(element.dataset.typewriterDeleteSpeed) || 50,
      pauseDuration: parseInt(element.dataset.typewriterPause) || 2000,
      color: element.dataset.typewriterColor || 'text-primary',
      wordAlternatives: wordAlternatives
    };
    
    new TypewriterEffect(element, options);
  });
});

export { TypewriterEffect };
