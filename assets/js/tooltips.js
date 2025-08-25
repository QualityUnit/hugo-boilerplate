// Tooltip functionality for mobile and desktop
document.addEventListener('DOMContentLoaded', function () {
  // Add CSS for mobile tooltips and improved positioning
  if (!document.getElementById('tooltip-styles')) {
    const style = document.createElement('style');
    style.id = 'tooltip-styles';
    style.textContent = `
      .tooltip-content.mobile-visible {
        opacity: 1 !important;
        pointer-events: auto !important;
      }
      
      /* Ensure tooltips don't overflow viewport */
      .tooltip-content {
        word-wrap: break-word;
        hyphens: auto;
        white-space: normal;
      }
      
      /* Smart positioning for tooltips near viewport edges */
      .tooltip-content.adjust-left {
        transform: translateX(-100%) translateY(-50%) !important;
        left: -8px !important;
        right: auto !important;
      }
      
      .tooltip-content.adjust-right {
        transform: translateX(0%) translateY(-50%) !important;
        left: calc(100% + 8px) !important;
        right: auto !important;
      }
      
      .tooltip-content.adjust-top {
        transform: translateX(-50%) translateY(-100%) !important;
        top: -8px !important;
        bottom: auto !important;
      }
    `;
    document.head.appendChild(style);
  }
  
  // Smart tooltip positioning
  function adjustTooltipPosition(tooltip) {
    if (!tooltip) return;
    
    const rect = tooltip.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Reset classes
    tooltip.classList.remove('adjust-left', 'adjust-right', 'adjust-top');
    
    // Check if tooltip goes off right edge
    if (rect.right > viewportWidth - 20) {
      tooltip.classList.add('adjust-left');
    }
    
    // Check if tooltip goes off left edge  
    if (rect.left < 20) {
      tooltip.classList.add('adjust-right');
    }
    
    // Check if tooltip goes off top edge
    if (rect.top < 20) {
      tooltip.classList.add('adjust-top');
    }
  }
  
  // Add hover listeners for desktop tooltips
  document.addEventListener('mouseenter', function(e) {
    if (!e.target || typeof e.target.closest !== 'function') return;
    const trigger = e.target.closest('.tooltip-trigger');
    if (trigger && window.innerWidth > 1023) {
      const group = trigger.closest('.group');
      const tooltip = group?.querySelector('.tooltip-content');
      if (tooltip) {
        // Small delay to ensure tooltip is rendered
        setTimeout(() => adjustTooltipPosition(tooltip), 10);
      }
    }
  }, true);
  
  function setupMobileTooltips() {
    const isMobile = window.innerWidth <= 1023;

    // Clear existing mobile tooltips
    document.querySelectorAll('.tooltip-content.mobile-visible').forEach(tip => {
      tip.classList.remove('mobile-visible');
    });
    
    // Handle click events for mobile tooltips
    document.addEventListener('click', function(e) {
      if (!isMobile) return;
      
      const trigger = e.target.closest('.tooltip-trigger');
      
      if (trigger) {
        const group = trigger.closest('.group');
        const content = group?.querySelector('.tooltip-content');
        
        // Close other tooltips
        document.querySelectorAll('.tooltip-content.mobile-visible').forEach(tip => {
          if (tip !== content) tip.classList.remove('mobile-visible');
        });
        
        // Toggle current tooltip
        if (content) {
          content.classList.toggle('mobile-visible');
          // Adjust position for mobile too
          if (content.classList.contains('mobile-visible')) {
            setTimeout(() => adjustTooltipPosition(content), 10);
          }
        }
        e.stopPropagation();
      } else {
        // Close all tooltips when clicking outside
        document.querySelectorAll('.tooltip-content.mobile-visible').forEach(tip => {
          tip.classList.remove('mobile-visible');
        });
      }
    });
  }
  
  // Initialize tooltips
  setupMobileTooltips();
  
  // Re-initialize on window resize
  window.addEventListener('resize', function() {
    setupMobileTooltips();
  });
});
