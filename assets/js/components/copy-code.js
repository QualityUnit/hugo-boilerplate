/**
 * Copy Code Component
 * Adds copy to clipboard functionality for code blocks
 */

document.addEventListener('DOMContentLoaded', function() {
    // Find all copy buttons
    const copyButtons = document.querySelectorAll('[data-copy-code]');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            // Find the code block within the same container
            const codeContainer = button.closest('.code-container');
            if (!codeContainer) return;
            
            const codeElement = codeContainer.querySelector('code');
            if (!codeElement) return;
            
            // Get the text content
            const codeText = codeElement.textContent || codeElement.innerText;
            
            // Copy to clipboard
            navigator.clipboard.writeText(codeText).then(() => {
                // Show success feedback
                showCopyFeedback(button, true);
            }).catch(err => {
                // Fallback for older browsers
                fallbackCopyTextToClipboard(codeText, button);
            });
        });
    });
});

/**
 * Fallback copy method for browsers that don't support navigator.clipboard
 */
function fallbackCopyTextToClipboard(text, button) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    
    // Avoid scrolling to bottom
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        // Note: document.execCommand('copy') is deprecated but kept for older browser compatibility
        // Modern browsers should use navigator.clipboard.writeText() which is handled in the main function
        const successful = document.execCommand('copy');
        showCopyFeedback(button, successful);
    } catch (err) {
        showCopyFeedback(button, false);
    }
    
    document.body.removeChild(textArea);
}

/**
 * Show visual feedback when code is copied
 */
function showCopyFeedback(button, success) {
    const originalContent = button.innerHTML;
    const originalClasses = button.className;
    
    // Get i18n strings from data attributes or fallback to English
    const copiedText = button.getAttribute('data-copied-text') || 'Copied!';
    const errorText = button.getAttribute('data-copy-error-text') || 'Copy error';
    const defaultText = button.getAttribute('data-copy-default-text') || 'Copy code';

    if (success) {
        // Show success state
        button.innerHTML = `
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
            </svg>
        `;
        button.className = originalClasses.replace('text-gray-500', 'text-green-500');
        button.title = copiedText;
    } else {
        // Show error state
        button.className = originalClasses.replace('text-gray-500', 'text-red-500');
        button.title = errorText;
    }
    
    // Reset after 2 seconds
    setTimeout(() => {
        button.innerHTML = originalContent;
        button.className = originalClasses;
        button.title = defaultText;
    }, 2000);
}