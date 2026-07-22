/**
 * Cookie Consent Script — two variants, selected by the rendered banner markup.
 *
 * LEGACY (default cookies-bar variant): the original non-blocking bottom bar +
 * 2-category settings modal. Behavior is intentionally byte-identical to the
 * pre-rework script: single cookie_consent_status cookie ('all' | 'necessary'),
 * Save maps "analytics on" -> 'all', vendor hooks keyed off analytics. Every
 * site that has not opted into the consent rework stays on this path.
 *
 * V2 (blocking variant, opt-in via [params.cookieConsent] blocking = true —
 * LiveAgent / PostAffiliatePro / FlowHunt / amicited): blocking modal +
 * 4-category settings. Adds the granular cookie_consent_v2 ("a1m0f1") next to
 * the legacy cookie (which is 'all' ONLY when every category is granted),
 * per-category Google Consent Mode updates, marketing-driven vendor hooks.
 * Skipped when window.consentCore owns Consent Mode (LiveAgent).
 *
 * The branch is detected from the data-consent-v2 attribute that only the
 * blocking cookies-bar variant renders on #cookie-consent-banner.
 */

const CookieManager = {
  set(name, value, days) {
    const date = new Date();
    date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
    // Optional Domain (window.__consentCookieDomain, set by cookies-bar.html from
    // [params.cookieConsent] cookieDomain) so the consent cookie is shared across
    // subdomains of one registrable domain. Unset (default) → host-only, unchanged.
    const dom = (typeof window.__consentCookieDomain === 'string' && window.__consentCookieDomain)
      ? `;Domain=${window.__consentCookieDomain}` : '';
    document.cookie = `${name}=${value};expires=${date.toUTCString()};path=/${dom}`;
  },

  get(name) {
    const nameEQ = `${name}=`;
    return document.cookie.split(';')
      .map(c => c.trim())
      .find(c => c.startsWith(nameEQ))
      ?.substring(nameEQ.length) || null;
  },

  delete(name) {
    this.set(name, "", -1);
  }
};

document.addEventListener('DOMContentLoaded', function() {
  const bannerEl = document.getElementById('cookie-consent-banner');
  if (bannerEl && bannerEl.hasAttribute('data-consent-v2')) {
    initConsentV2();
  } else {
    initConsentLegacy();
  }
});

function initConsentLegacy() {
  const banner = document.getElementById('cookie-consent-banner');
  const modal = document.getElementById('cookie-settings-modal');
  const analyticsCheckbox = document.getElementById('analytics-cookies');
  const consentStatus = CookieManager.get('cookie_consent_status');

  if (banner) {
    banner.removeAttribute('style');
  }

  if (consentStatus) {
    hideBanner();
  } else {
    showBanner();
  }

  if (consentStatus === 'all' && analyticsCheckbox) {
    analyticsCheckbox.checked = true;
    updateAnalyticsConsent(true);
  }

  document.addEventListener('click', function(event) {
    if (event.target.closest('[data-cookie-consent="accept-all"]')) {
      event.preventDefault();
      setConsent('all');
      hideBanner();
    }

    if (event.target.closest('[data-cookie-consent="accept-necessary"]')) {
      event.preventDefault();
      setConsent('necessary');
      hideBanner();
    }

    if (event.target.closest('[data-cookie-consent="settings"]')) {
      event.preventDefault();
      modal?.classList.remove('hidden');
    }

    if (event.target.closest('[data-cookie-settings-close]')) {
      event.preventDefault();
      modal?.classList.add('hidden');
    }

    if (event.target.closest('[data-cookie-settings-save]')) {
      event.preventDefault();
      const allowAnalytics = analyticsCheckbox?.checked || false;
      setConsent(allowAnalytics ? 'all' : 'necessary');
      modal?.classList.add('hidden');
      hideBanner();
    }
  });

  function hideBanner() {
    if (banner) {
      banner.removeAttribute('style');
      banner.classList.add('cookie-hidden');
      banner.style.display = 'none';
      banner.style.visibility = 'hidden';
      banner.style.opacity = '0';
    }
  }

  function showBanner() {
    if (banner) {
      banner.classList.remove('cookie-hidden');
      banner.style.display = 'block';
      banner.style.visibility = 'visible';
      banner.style.opacity = '1';
    }
  }

  function setConsent(level) {
    CookieManager.set('cookie_consent_status', level, 365);
    updateAnalyticsConsent(level === 'all');
    console.log('Cookie consent set to:', level); // Pre debugovanie
    
    // If user accepts all cookies, set YouTube GDPR consent too
    if (level === 'all' && window.flowhuntMedia && window.flowhuntMedia.video && typeof window.flowhuntMedia.video.setGdprConsent === 'function') {
      window.flowhuntMedia.video.setGdprConsent(true);
    }
  }

  function updateAnalyticsConsent(allowed) {
    // Consent Mode update: skip when consent-core.js is present (LiveAgent) — it owns
    // the full 7-parameter update and fires it on the same banner click. Firing this
    // partial 4-parameter update too would duplicate it in the dataLayer. Sites without
    // consent-core (e.g. PAP / FlowHunt) still fire it here as before.
    if (typeof window.gtag === 'function' && !window.consentCore) {
      window.gtag('consent', 'update', {
        'analytics_storage': allowed ? 'granted' : 'denied',
        'ad_storage': allowed ? 'granted' : 'denied',
        'ad_user_data': allowed ? 'granted' : 'denied',
        'ad_personalization': allowed ? 'granted' : 'denied'
      });
    }
    
    // Update Capterra consent
    if (typeof window.updateCapterraConsent === 'function') {
      window.updateCapterraConsent(allowed);
    }

    // Update Meta Pixel consent
    if (typeof window.updateMetaPixelConsent === 'function') {
      window.updateMetaPixelConsent(allowed);
    }
  }
}

function initConsentV2() {
  const banner = document.getElementById('cookie-consent-banner');
  const modal = document.getElementById('cookie-settings-modal');
  const DAYS = 365;

  function readGranular() {
    const m = /^a([01])m([01])f([01])$/.exec(CookieManager.get('cookie_consent_v2') || '');
    return m ? { analytics: m[1] === '1', marketing: m[2] === '1', functional: m[3] === '1' } : null;
  }

  function writeGranular(c) {
    CookieManager.set('cookie_consent_v2',
      'a' + (c.analytics ? 1 : 0) + 'm' + (c.marketing ? 1 : 0) + 'f' + (c.functional ? 1 : 0), DAYS);
  }

  function switchOn(id) {
    const el = document.getElementById(id);
    return !!(el && el.checked);
  }

  function setSwitch(id, on) {
    const el = document.getElementById(id);
    if (el) el.checked = !!on;
  }

  // Stored choice: the granular cookie wins; the legacy cookie maps
  // all → everything granted, necessary → everything denied.
  function currentChoice() {
    const g = readGranular();
    if (g) return g;
    const s = CookieManager.get('cookie_consent_status');
    if (s === 'all') return { analytics: true, marketing: true, functional: true };
    if (s === 'necessary') return { analytics: false, marketing: false, functional: false };
    return null;
  }

  function syncSwitches() {
    const c = currentChoice() || { analytics: false, marketing: false, functional: false };
    setSwitch('analytics-cookies', c.analytics);
    setSwitch('marketing-cookies', c.marketing);
    setSwitch('functional-cookies', c.functional);
  }

  if (banner) {
    banner.removeAttribute('style');
  }

  const stored = currentChoice();

  if (stored) {
    hideBanner();
    syncSwitches();
    updateVendorConsent(stored);
  } else {
    showBanner();
  }

  document.addEventListener('click', function(event) {
    if (event.target.closest('[data-cookie-consent="accept-all"]')) {
      event.preventDefault();
      setConsent({ analytics: true, marketing: true, functional: true });
      hideBanner();
    }

    if (event.target.closest('[data-cookie-consent="accept-necessary"]')) {
      event.preventDefault();
      setConsent({ analytics: false, marketing: false, functional: false });
      hideBanner();
    }

    if (event.target.closest('[data-cookie-consent="settings"]')) {
      event.preventDefault();
      syncSwitches();
      modal?.classList.remove('hidden');
      // The banner is a blocking overlay — never stack the two dialogs.
      hideBanner();
    }

    if (event.target.closest('[data-cookie-settings-close]')) {
      event.preventDefault();
      modal?.classList.add('hidden');
      // Cancel/X without any stored decision: bring the (blocking) banner back.
      if (!currentChoice()) {
        showBanner();
      }
    }

    if (event.target.closest('[data-cookie-settings-save]')) {
      event.preventDefault();
      setConsent({
        analytics: switchOn('analytics-cookies'),
        marketing: switchOn('marketing-cookies'),
        functional: switchOn('functional-cookies')
      });
      modal?.classList.add('hidden');
      hideBanner();
    }
  });

  function hideBanner() {
    if (banner) {
      banner.removeAttribute('style');
      banner.classList.add('cookie-hidden');
      banner.style.display = 'none';
      banner.style.visibility = 'hidden';
      banner.style.opacity = '0';
    }
  }

  function showBanner() {
    if (banner) {
      banner.classList.remove('cookie-hidden');
      banner.style.display = 'block';
      banner.style.visibility = 'visible';
      banner.style.opacity = '1';
    }
  }

  function setConsent(c) {
    const all = c.analytics && c.marketing && c.functional;
    CookieManager.set('cookie_consent_status', all ? 'all' : 'necessary', DAYS);
    writeGranular(c);
    updateVendorConsent(c);
    console.log('Cookie consent set:',
      'analytics=' + c.analytics + ' marketing=' + c.marketing + ' functional=' + c.functional);

    // If any tracking category is granted, set YouTube GDPR consent too
    if ((c.analytics || c.marketing) && window.flowhuntMedia && window.flowhuntMedia.video &&
        typeof window.flowhuntMedia.video.setGdprConsent === 'function') {
      window.flowhuntMedia.video.setGdprConsent(true);
    }
  }

  function updateVendorConsent(c) {
    // Consent Mode update: skip when consent-core.js is present (LiveAgent) — it owns
    // the full update and fires it on the same banner clicks. Firing this one too
    // would duplicate it in the dataLayer. Sites without consent-core (e.g. PAP /
    // FlowHunt) get the per-category update here.
    if (typeof window.gtag === 'function' && !window.consentCore) {
      window.gtag('consent', 'update', {
        'analytics_storage': c.analytics ? 'granted' : 'denied',
        'ad_storage': c.marketing ? 'granted' : 'denied',
        'ad_user_data': c.marketing ? 'granted' : 'denied',
        'ad_personalization': c.marketing ? 'granted' : 'denied',
        'functionality_storage': c.functional ? 'granted' : 'denied',
        'personalization_storage': c.functional ? 'granted' : 'denied',
        'security_storage': 'granted'
      });
      window.gtag('set', 'ads_data_redaction', !c.marketing);
    }

    // Marketing vendors follow the MARKETING category (previously keyed off analytics).
    if (typeof window.updateCapterraConsent === 'function') {
      window.updateCapterraConsent(c.marketing);
    }

    if (typeof window.updateMetaPixelConsent === 'function') {
      window.updateMetaPixelConsent(c.marketing);
    }
  }
}
