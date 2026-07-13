/**
 * consent-modal.js — small helper for the cookie-settings modal + the US
 * "Your Privacy Choices" footer link. Banner + categories are unchanged
 * (Necessary + Analytics); the theme cookie-consent.js still opens/closes the modal,
 * handles the buttons and writes consent (data-cookie-settings-save). We only:
 *   - sync the Analytics switch from window.__consent whenever the modal opens, so
 *     US opt-out visitors correctly see it ON by default (derive() grants region row),
 *   - reveal the footer "Your Privacy Choices" link ONLY for country = US (CCPA/CPRA);
 *     it opens the same modal via data-cookie-consent="settings".
 *
 * GPC honoring is a separate (mandatory) piece for consent-core.
 */
(function () {
  if (window.consentModalLoaded) return;
  window.consentModalLoaded = true;

  function byId(id) { return document.getElementById(id); }

  // Initialize the category switches from the live consent state (set by consent-core).
  function syncFromConsent() {
    var c = window.__consent || {};
    var a = byId('analytics-cookies');
    var m = byId('marketing-cookies');
    var f = byId('functional-cookies');
    if (a) a.checked = !!c.analytics;
    if (m) m.checked = !!c.marketing;
    if (f) f.checked = !!c.functional;
  }

  // When the modal opens (theme removes `hidden`), re-init from current consent.
  // Also hide the (blocking, fullscreen) banner while the settings modal is open —
  // the theme JS leaves it shown, so the two dialogs would stack. Class-based via
  // html.cc-settings-open (custom.css), consistent with cc-hide/cc-show: no direct
  // element poking, so it can't fight the theme's inline styles. Cancel/X without a
  // decision removes the class and the banner comes back; on Save the theme's own
  // hideBanner() keeps it hidden for good.
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-cookie-consent="settings"]')) {
      setTimeout(syncFromConsent, 0);
      document.documentElement.classList.add('cc-settings-open');
    }
    if (e.target.closest('[data-cookie-settings-close]') ||
        e.target.closest('[data-cookie-settings-save]')) {
      document.documentElement.classList.remove('cc-settings-open');
    }
  });

  // Footer "Your Privacy Choices" link — show ONLY for US visitors (country-level;
  // CCPA/CPRA + other US state opt-out laws). Hidden by default in the markup.
  function gateFooterLink() {
    var link = byId('your-privacy-choices');
    if (!link) return;
    var geo = window.__geo;
    var isUS = geo && geo.country && String(geo.country).toUpperCase() === 'US';
    if (isUS) { link.classList.remove('hidden'); link.classList.add('inline-flex'); }
  }
  document.addEventListener('geo:ready', gateFooterLink);

  function init() { syncFromConsent(); gateFooterLink(); }
  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
