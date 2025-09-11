/**
 * Universal Item Toggle Component
 * Handles show more/less functionality for any type of item lists (tags, resources, etc.)
 */

function initItemToggle() {
    // Add event listeners to show more buttons
    document.querySelectorAll('.show-more-btn').forEach(function(button) {
        button.addEventListener('click', function() {
            const containerId = this.getAttribute('data-container');
            const containerType = this.getAttribute('data-container-type') || 'items';
            showMoreItems(containerId, containerType);
        });
    });
    
    // Add event listeners to show less buttons
    document.querySelectorAll('.show-less-btn').forEach(function(button) {
        button.addEventListener('click', function() {
            const containerId = this.getAttribute('data-container');
            const containerType = this.getAttribute('data-container-type') || 'items';
            showLessItems(containerId, containerType);
        });
    });
}

function showMoreItems(containerId, containerType) {
    const container = document.getElementById(`${containerType}-container-${containerId}`);
    if (!container) return;
    
    // Find hidden items based on container type
    const hiddenItemClass = getHiddenItemClass(containerType);
    const hiddenItems = container.querySelectorAll(hiddenItemClass);
    const showMoreBtn = container.querySelector('.show-more-btn');
    const showLessBtn = container.querySelector('.show-less-btn');
    
    // Show hidden items
    hiddenItems.forEach(function(item) {
        item.classList.remove('hidden');
    });
    
    // Hide show more button, show show less button
    if (showMoreBtn) showMoreBtn.classList.add('hidden');
    if (showLessBtn) showLessBtn.classList.remove('hidden');
}

function showLessItems(containerId, containerType) {
    const container = document.getElementById(`${containerType}-container-${containerId}`);
    if (!container) return;
    
    // Find hidden items based on container type
    const hiddenItemClass = getHiddenItemClass(containerType);
    const hiddenItems = container.querySelectorAll(hiddenItemClass);
    const showMoreBtn = container.querySelector('.show-more-btn');
    const showLessBtn = container.querySelector('.show-less-btn');
    
    // Hide additional items
    hiddenItems.forEach(function(item) {
        item.classList.add('hidden');
    });
    
    // Hide show less button, show show more button
    if (showLessBtn) showLessBtn.classList.add('hidden');
    if (showMoreBtn) showMoreBtn.classList.remove('hidden');
}

function toggleItems(containerId, containerType) {
    // Keep this for backward compatibility
    const container = document.getElementById(`${containerType}-container-${containerId}`);
    if (!container) return;
    
    const hiddenItemClass = getHiddenItemClass(containerType);
    const hiddenItems = container.querySelectorAll(hiddenItemClass);
    const showMoreBtn = container.querySelector('.show-more-btn');
    const showLessBtn = container.querySelector('.show-less-btn');
    
    hiddenItems.forEach(function(item) {
        item.classList.toggle('hidden');
    });
    
    if (showMoreBtn) showMoreBtn.classList.toggle('hidden');
    if (showLessBtn) showLessBtn.classList.toggle('hidden');
}

function getHiddenItemClass(containerType) {
    const classMap = {
        'tags': '.tag-hidden',
        'resources': '.resource-hidden',
        'items': '.item-hidden' // default fallback
    };
    
    return classMap[containerType] || classMap['items'];
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initItemToggle);

// Re-initialize for dynamically added content
window.reinitItemToggle = initItemToggle;

// Backward compatibility aliases
window.reinitTagToggle = initItemToggle;
window.reinitResourceToggle = initItemToggle;
