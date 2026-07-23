/**
 * consent-core.js — consent bridge for the (unchanged) theme cookie banner.
 *
 * Phase 1 keeps the ORIGINAL theme banner + theme cookie-consent.js (Accept all /
 * Accept necessary / Settings, legacy `cookie_consent_status` = 'all' | 'necessary').
 * This bridge adds the region-aware + granular layer the reworked tracking partials
 * need, WITHOUT changing the banner UI (the 3-button / Marketing redesign is Phase 2):
 *
 *  - derives granular consent (analytics / marketing) from the legacy cookie + the
 *    geo `consent_region` (eu/row) default,
 *  - sets Google Consent Mode v2 (all 7 params),
 *  - exposes window.__consent + fires a `consentUpdate` event that every reworked
 *    tracking partial subscribes to,
 *  - for non-EU (row) with no prior choice: auto-grants (sets cookie_consent_status='all')
 *    and hides the theme banner,
 *  - observes the theme banner's own buttons (it doesn't handle them) and re-applies,
 *  - exposes window.consentCore.grantAll() for implicit consent on trial signup.
 *
 * Category → Consent Mode mapping (per spec):
 *   Necessary  → security_storage (always granted) + Chat + dataLayer events
 *   Analytics  → analytics_storage           (GA4, FlowHunt, Grafana/Matomo)
 *   Marketing  → ad_storage + ad_user_data + ad_personalization
 *                (Meta Pixel, Google Ads, Capterra, Post Affiliate Pro, crmsor)
 *   Functional → functionality_storage + personalization_storage (preferences)
 *
 * 4-category storage (issue #4195): the settings modal has Analytics + Marketing +
 * Functional switches, which the legacy 2-state cookie cannot express (the theme's
 * Save maps "analytics on" to 'all', silently granting marketing). Granular choices
 * are persisted in the `cookie_consent_v2` cookie ("a1m0f1"), written HERE on the
 * banner/settings clicks. It wins over the legacy cookie in derive() and in the
 * consent DEFAULT (google-analytics-consent.html); the theme-owned legacy cookie
 * remains only as the "a choice was made" marker for banner show/hide.
 */
(function () {
  if (window.consentCoreLoaded) return;
  window.consentCoreLoaded = true;

  var LEGACY = 'cookie_consent_status'; // owned by the theme banner
  var GRANULAR = 'cookie_consent_v2';   // owned by consent-core — "a<0|1>m<0|1>f<0|1>"
  var DAYS = 365;

  function getCookie(name) {
    var m = document.cookie.match('(?:^|; )' + name + '=([^;]*)');
    return m ? decodeURIComponent(m[1]) : null;
  }
  function setCookie(name, value, days) {
    var exp = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = name + '=' + encodeURIComponent(value) +
      '; expires=' + exp + '; path=/; SameSite=Lax';
  }

  function region() {
    var r = getCookie('consent_region');
    return (r === 'eu' || r === 'row') ? r : null; // null = not yet known
  }

  function parseGranular(v) {
    var m = /^a([01])m([01])f([01])$/.exec(v || '');
    return m ? { a: m[1] === '1', m: m[2] === '1', f: m[3] === '1' } : null;
  }
  function writeGranular(a, m, f) {
    setCookie(GRANULAR, 'a' + (a ? 1 : 0) + 'm' + (m ? 1 : 0) + 'f' + (f ? 1 : 0), DAYS);
  }
  function switchOn(id) {
    var el = document.getElementById(id);
    return !!(el && el.checked);
  }

  // Derive per-category consent: granular cookie wins, then the legacy 2-state
  // cookie (pre-v2 visitors), then the region default.
  function derive() {
    var g = parseGranular(getCookie(GRANULAR));
    if (g) return { a: g.a, m: g.m, f: g.f, src: 'choice' };
    var s = getCookie(LEGACY);
    if (s === 'all') return { a: true, m: true, f: true, src: 'choice' };
    if (s === 'necessary') return { a: false, m: false, f: false, src: 'choice' };
    var r = region();
    if (r === 'row') return { a: true, m: true, f: true, src: 'region-row' };
    return { a: false, m: false, f: false, src: r === 'eu' ? 'region-eu' : 'region-unknown' };
  }

  // apply(fromInteraction): re-derive consent, expose it, and fire the internal
  // consentUpdate event. The Google Consent Mode `update` is pushed ONLY on a real
  // interaction (banner click / programmatic set/grantAll) — NOT on page-load init.
  // On init a stored choice is already surfaced as a denied-default → granted-UPDATE
  // by STEP 1 inline in google-analytics-consent.html (runs before the GTM loader);
  // re-emitting the same update from here on every load was a duplicate `consent
  // update` (+ `ads_data_redaction`) in the dataLayer. window.__consent and the
  // consentUpdate event still fire on init so tracking partials react regardless.
  function apply(fromInteraction) {
    var c = derive();
    window.__consent = {
      necessary: true, analytics: c.a, marketing: c.m, functional: c.f,
      region: region(), source: c.src
    };

    if (fromInteraction && c.src === 'choice' && typeof window.gtag === 'function') {
      window.gtag('consent', 'update', {
        'analytics_storage': c.a ? 'granted' : 'denied',
        'ad_storage': c.m ? 'granted' : 'denied',
        'ad_user_data': c.m ? 'granted' : 'denied',
        'ad_personalization': c.m ? 'granted' : 'denied',
        'functionality_storage': c.f ? 'granted' : 'denied',
        'personalization_storage': c.f ? 'granted' : 'denied',
        'security_storage': 'granted'
      });
      window.gtag('set', 'ads_data_redaction', !c.m);
    }

    if ((c.a || c.m) && window.flowhuntMedia && window.flowhuntMedia.video &&
        typeof window.flowhuntMedia.video.setGdprConsent === 'function') {
      window.flowhuntMedia.video.setGdprConsent(true);
    }

    try {
      document.dispatchEvent(new CustomEvent('consentUpdate', { detail: window.__consent }));
    } catch (e) { /* non-critical */ }

    if (window.console && console.debug) {
      console.debug('[consent-core] analytics=' + c.a + ' marketing=' + c.m +
        ' functional=' + c.f + ' region=' + window.__consent.region + ' source=' + c.src);
    }
  }

  // Hide via the same <html> class geo-consent.js uses (custom.css resolves it).
  // Single, order-independent mechanism — no direct banner-element poking.
  function hideThemeBanner() {
    document.documentElement.classList.add('cc-hide');
    document.documentElement.classList.remove('cc-show');
  }

  // Public API — programmatic consent (implicit consent on trial signup:
  // crm-installer-tracking.html / trial-forms.js).
  window.consentCore = {
    set: function (analytics, marketing, functional) {
      if (functional === undefined) functional = marketing; // pre-v2 2-arg callers
      writeGranular(analytics, marketing, functional);
      setCookie(LEGACY, (analytics && marketing && functional) ? 'all' : 'necessary', DAYS);
      apply(true); // programmatic consent = interaction → emit the Consent Mode update
    },
    grantAll: function () {
      writeGranular(true, true, true);
      setCookie(LEGACY, 'all', DAYS);
      apply(true); // implicit consent (trial signup) → emit the update
      hideThemeBanner();
    },
    get: function () { return window.__consent || null; }
  };

  function maybeAutoGrantRow() {
    // Non-EU + no explicit choice → only HIDE the banner. We deliberately do NOT
    // persist cookie_consent_status='all' here: tracking is granted at runtime by
    // derive() (region 'row' → analytics+marketing), so the cookie would be redundant.
    // Worse, it is indistinguishable from a real user choice and survives a region
    // change — a visitor first seen as 'row' would then keep full tracking and get NO
    // banner if their region later resolves to 'eu' (GDPR issue). Leaving the cookie
    // unset means derive() re-evaluates from the live region every load.
    if (!getCookie(LEGACY) && region() === 'row') {
      hideThemeBanner();
    }
  }

  function run() {
    maybeAutoGrantRow();
    apply();

    // Region may be unknown on first visit until the geo fetch resolves.
    document.addEventListener('geo:ready', function () {
      if (!getCookie(LEGACY)) { maybeAutoGrantRow(); apply(); }
    });

    // The theme's own cookie-consent.js handles the banner buttons + sets the
    // LEGACY cookie — but its 2-state value can't express partial choices (its
    // Save maps "analytics on" to 'all'). Persist the GRANULAR cookie here on the
    // same clicks; derive() and the GA consent DEFAULT prefer it. Then re-derive
    // (fire consentUpdate) a tick later so services react within the session.
    document.addEventListener('click', function (e) {
      if (e.target.closest('[data-cookie-consent="accept-all"]')) {
        writeGranular(true, true, true);
      }
      if (e.target.closest('[data-cookie-consent="accept-necessary"]')) {
        writeGranular(false, false, false);
      }
      if (e.target.closest('[data-cookie-settings-save]')) {
        writeGranular(switchOn('analytics-cookies'), switchOn('marketing-cookies'),
          switchOn('functional-cookies'));
      }
      if (e.target.closest('[data-cookie-consent], [data-cookie-settings-save]')) {
        setTimeout(function () { apply(true); }, 60); // banner click = interaction → emit update
      }
    });
  }

  if (document.readyState === 'complete') run();
  else document.addEventListener('DOMContentLoaded', run);
})();
