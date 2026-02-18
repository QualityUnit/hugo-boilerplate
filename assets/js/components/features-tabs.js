/**
 * Features Tabs Component
 * Handles tab switching and lazy image loading for features-with-intro-and-tabs sections.
 */

function loadLazyImages(panel) {
    panel.querySelectorAll('img.lazy-image[data-src]').forEach(function(image) {
        var picture = image.closest('picture');
        if (picture) {
            picture.querySelectorAll('source[data-srcset]').forEach(function(source) {
                source.srcset = source.dataset.srcset;
                source.removeAttribute('data-srcset');
            });
        }
        image.src = image.dataset.src;
        image.removeAttribute('data-src');
        image.onload = function() {
            image.classList.add('loaded');
            if (picture) {
                picture.classList.add('loaded');
            }
        };
    });
}

function initFeaturesTabs() {
    var tabButtons = document.querySelectorAll('button[data-tabs-group]');

    tabButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            var tabId = this.getAttribute('data-tab-id');
            var tabsGroup = this.getAttribute('data-tabs-group');

            // Hide all panels in this group
            document.querySelectorAll('div[data-tabs-group="' + tabsGroup + '"]').forEach(function(panel) {
                panel.classList.add('hidden');
            });

            // Show the selected panel
            var activePanel = document.querySelector('div[data-tabs-group="' + tabsGroup + '"][data-panel-id="' + tabId + '"]');
            if (activePanel) {
                activePanel.classList.remove('hidden');
                loadLazyImages(activePanel);
            }

            // Reset all tab button styles in this group
            document.querySelectorAll('button[data-tabs-group="' + tabsGroup + '"]').forEach(function(btn) {
                btn.classList.remove('border-brand', 'product-primary');
                btn.classList.add('border-transparent', 'text-tertiary', 'hover:border-brand', 'hover:product-primary');
            });

            // Style the active tab
            this.classList.remove('border-transparent', 'text-tertiary', 'hover:border-brand', 'hover:product-primary');
            this.classList.add('border-brand', 'product-primary');
        });
    });
}

// Initialize when DOM is ready, or immediately if already loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFeaturesTabs);
} else {
    initFeaturesTabs();
}

// Export for external use
window.loadLazyImages = loadLazyImages;
window.reinitFeaturesTabs = initFeaturesTabs;
