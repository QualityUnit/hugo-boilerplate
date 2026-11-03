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
    `;
    document.head.appendChild(style);
  }

  // Tap-to-toggle below 1024px, hover/focus (CSS group-hover/focus-within) above.
  // Checked on every event, so a resize or rotation switches the behaviour.
  const tapQuery = window.matchMedia('(max-width: 1023px)');
  const cardQuery = window.matchMedia('(max-width: 767px)');

  let openScrollY = 0;

  function placeCard(tooltip, trigger) {
    const gap = 8;
    const r = trigger.getBoundingClientRect();
    const h = tooltip.offsetHeight;
    if (r.bottom + gap + h <= window.innerHeight - gap) {
      tooltip.style.top = (r.bottom + gap) + 'px';
      tooltip.style.bottom = 'auto';
    } else {
      tooltip.style.top = 'auto';
      tooltip.style.bottom = Math.max(gap, window.innerHeight - r.top + gap) + 'px';
    }
  }

  function clearPosition(tooltip) {
    tooltip.style.top = '';
    tooltip.style.bottom = '';
    tooltip.style.marginLeft = '';
  }

  // Keep the tooltip inside the viewport: shift it sideways, and flip a top/bottom
  // tooltip to the other side of the icon when it does not fit vertically.
  function adjustTooltipPosition(tooltip, trigger) {
    if (!tooltip) return;
    clearPosition(tooltip);

    // Phones: full-width fixed card; only move it next to the tapped icon
    // (below it, or above when there is no room), away from the bottom corner widgets.
    if (cardQuery.matches) {
      if (trigger) placeCard(tooltip, trigger);
      return;
    }

    const edge = 8;
    const position = tooltip.dataset.position || 'top';
    let rect = tooltip.getBoundingClientRect();
    if (position === 'top' && rect.top < edge) {
      tooltip.style.bottom = 'auto';
      tooltip.style.top = 'calc(100% + 0.5rem)';
    } else if (position === 'bottom' && rect.bottom > window.innerHeight - edge) {
      tooltip.style.top = 'auto';
      tooltip.style.bottom = 'calc(100% + 0.5rem)';
    }

    rect = tooltip.getBoundingClientRect();
    const viewportWidth = document.documentElement.clientWidth;
    if (rect.left < edge) {
      tooltip.style.marginLeft = (edge - rect.left) + 'px';
    } else if (rect.right > viewportWidth - edge) {
      tooltip.style.marginLeft = (viewportWidth - edge - rect.right) + 'px';
    }
  }

  function tooltipFor(el) {
    const wrapper = el.closest('[data-tooltip-wrap]');
    return wrapper ? wrapper.querySelector('.tooltip-content') : null;
  }

  function closeAll(except) {
    document.querySelectorAll('.tooltip-content.mobile-visible').forEach(tip => {
      if (tip !== except) tip.classList.remove('mobile-visible');
    });
  }

  // Desktop: adjust position when the tooltip appears on hover (label or icon) or keyboard focus
  // (also 768-1023px: tablets with a mouse hover there too). Not on phones: the card
  // is placed on tap, and emulated mouse events must not undo that.
  function onReveal(e) {
    if (cardQuery.matches) return;
    if (!e.target || typeof e.target.closest !== 'function') return;
    const tooltip = tooltipFor(e.target);
    // Laid out even while transparent, so it can be measured right away
    if (tooltip) adjustTooltipPosition(tooltip);
  }
  document.addEventListener('mouseenter', onReveal, true);
  document.addEventListener('focusin', onReveal);

  // Mobile/tablet: one listener for the whole page. The trigger is the <button>,
  // so a click, a tap and Enter/Space on a focused button all land here.
  document.addEventListener('click', function (e) {
    if (!tapQuery.matches) return;

    const trigger = e.target.closest('.tooltip-trigger');
    if (!trigger) {
      // Close all tooltips when clicking outside
      closeAll();
      return;
    }

    const content = tooltipFor(trigger);
    closeAll(content);
    if (content) {
      content.classList.toggle('mobile-visible');
      if (content.classList.contains('mobile-visible')) {
        openScrollY = window.scrollY;
        adjustTooltipPosition(content, trigger);
      }
    }
  });

  // The phone card is fixed to the viewport, so it would drift away from its icon on scroll
  // (a small threshold keeps it open through URL-bar jitter and focus scroll-into-view).
  window.addEventListener('scroll', function () {
    if (cardQuery.matches && Math.abs(window.scrollY - openScrollY) > 24) closeAll();
  }, { passive: true });

  // Escape closes an open tooltip (WCAG 1.4.13: dismissible)
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    closeAll();
    // Desktop shows it via :focus-within, so drop focus from the trigger
    const active = document.activeElement;
    if (active && active.classList && active.classList.contains('tooltip-trigger')) active.blur();
  });

  // Crossing a breakpoint (e.g. phone rotation): drop open tooltips and stale adjustments.
  // Not on every resize: mobile browsers fire resize while the URL bar hides on scroll.
  function reset() {
    closeAll();
    document.querySelectorAll('.tooltip-content').forEach(tip => {
      clearPosition(tip);
    });
  }
  tapQuery.addEventListener('change', reset);
  cardQuery.addEventListener('change', reset);
});
