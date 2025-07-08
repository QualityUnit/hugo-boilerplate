/**
 * Tag Toggle Component
 * Handles show more/less functionality for tag lists
 */

function initTagToggle() {
    // Add event listeners to all show more/less buttons
    document.querySelectorAll('.show-more-btn, .show-less-btn').forEach(function(button) {
        button.addEventListener('click', function() {
            const containerId = this.getAttribute('data-container');
            toggleTags(containerId);
        });
    });
}

function toggleTags(containerId) {
    const container = document.getElementById('tags-container-' + containerId);
    if (!container) return;
    
    const hiddenTags = container.querySelectorAll('.tag-hidden');
    const showMoreBtn = container.querySelector('.show-more-btn');
    const showLessBtn = container.querySelector('.show-less-btn');
    
    hiddenTags.forEach(function(tag) {
        tag.classList.toggle('hidden');
    });
    
    if (showMoreBtn) showMoreBtn.classList.toggle('hidden');
    if (showLessBtn) showLessBtn.classList.toggle('hidden');
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initTagToggle);

// Re-initialize for dynamically added content
window.reinitTagToggle = initTagToggle;
