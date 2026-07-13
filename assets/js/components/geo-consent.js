/**
 * geo-consent.js — client-side visitor-country detection for cookie consent.
 *
 * Fetches the edge geo endpoint (a standalone CloudFront function that returns
 * {"country":"XX"} based on the viewer IP — see
 * QualityUnit/infrastructure-issues#2310), derives a consent region
 * (eu | row), stores it in the `consent_region` cookie, exposes `window.__geo`,
 * dispatches a `geo:ready` event, and OWNS cookie-banner visibility.
 *
 * Banner model (fixes the flash reported for already-consented + non-EU visitors):
 * the banner is CSS-hidden by default (#cookie-consent-banner{display:none} in
 * custom.css), so it never paints before JS runs. The theme cookie-consent.js
 * unconditionally shows it for visitors without a prior choice in its
 * DOMContentLoaded handler — this component's handler runs in the SAME synchronous
 * DCL dispatch right after it (no paint in between), so we re-hide and then reveal
 * the banner ONLY once the region is known to require consent:
 *   - prior choice (cookie_consent_status set) → never show
 *   - region row (non-EU)                       → never show (consent-core auto-grants)
 *   - region eu, or unknown after the fetch     → show (GDPR-safe)
 * EU visitors therefore see the banner a beat later (after geo resolves) — an
 * accepted trade-off for eliminating the flash.
 *
 * Standalone component: built by `gulp components` to static/js/geo-consent.js
 * and loaded via layouts/partials/geo-consent.html. The endpoint is configurable
 * through the script tag's `data-geo-endpoint` attribute.
 *
 * Region source priority: ?geo_country override → `geo_country` cookie (set at the
 * edge by the CloudFront viewer-response function, scripts/edge/geo-country-cookie.js
 * — authoritative, no network) → geo endpoint fetch (fallback). The edge cookie path
 * removes the geo round-trip entirely once that function is deployed.
 *
 * Test override:  ?geo_country=US  → force a country (skips the network call).
 */
(function () {
  if (window.geoConsentLoaded) return;
  window.geoConsentLoaded = true;

  // OPT-IN jurisdictions — a cookie-consent banner is required and non-essential
  // trackers must stay OFF until the visitor consents. Region value stays 'eu'
  // (= the "opt-in / consent-required" bucket), even for the non-European members:
  //   EU-27 + EEA (IS, LI, NO) + UK (GB) + Switzerland (CH, nDSG)
  //   + Canada (CA, Québec Law 25 — country-only geo can't isolate Québec, so all CA)
  //   + Brazil (BR, LGPD).
  // NOT here: US — opt-OUT model (trackers may run by default; the footer
  // "Your Privacy Choices" link handles opt-out), so US falls through to 'row'.
  var CONSENT_REQUIRED = {
    AT: 1, BE: 1, BG: 1, HR: 1, CY: 1, CZ: 1, DK: 1, EE: 1, FI: 1, FR: 1,
    DE: 1, GR: 1, HU: 1, IE: 1, IT: 1, LV: 1, LT: 1, LU: 1, MT: 1, NL: 1,
    PL: 1, PT: 1, RO: 1, SK: 1, SI: 1, ES: 1, SE: 1, IS: 1, LI: 1, NO: 1,
    GB: 1, CH: 1, CA: 1, BR: 1
  };

  // Read the endpoint at script-execution time (document.currentScript is null
  // inside the later DOMContentLoaded handler).
  var script = document.currentScript;
  var ENDPOINT = (script && script.getAttribute('data-geo-endpoint')) ||
    'https://geo.liveagent.com/';
  var COOKIE_DAYS = 365;

  function getCookie(name) {
    var m = document.cookie.match('(?:^|; )' + name + '=([^;]*)');
    return m ? decodeURIComponent(m[1]) : null;
  }

  function setCookie(name, value, days) {
    var exp = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = name + '=' + encodeURIComponent(value) +
      '; expires=' + exp + '; path=/; SameSite=Lax';
  }

  function regionFor(country) {
    if (!country) return null;
    return CONSENT_REQUIRED[country.toUpperCase()] ? 'eu' : 'row';
  }

  // Visibility is expressed as a class on <html>, resolved by CSS (custom.css):
  //   html.cc-hide #banner { display:none !important }  — beats the theme's inline
  //                                                        display:block regardless of
  //                                                        script execution order
  //   html.cc-show #banner { display:block }            — non-important, so the theme's
  //                                                        .cookie-hidden (accept) still wins
  // We never touch the banner element directly, so this works even before the element
  // is parsed and no longer depends on running AFTER the theme's handler.
  function showBanner() {
    var c = document.documentElement.classList;
    c.add('cc-show'); c.remove('cc-hide');
  }
  function hideBanner() {
    var c = document.documentElement.classList;
    c.add('cc-hide'); c.remove('cc-show');
  }

  // Decide banner visibility for a known region. Never touches tracking consent —
  // visibility only. Call with null only AFTER the fetch resolved/failed (unknown
  // region → show, GDPR-safe); during the pending phase the banner stays hidden.
  function applyBanner(region) {
    if (getCookie('cookie_consent_status')) { hideBanner(); return; } // already chose
    if (region === 'row') { hideBanner(); return; }                  // non-EU → auto-grant
    showBanner();                                                    // eu / unknown
  }

  function publish(country, region, source) {
    window.__geo = { country: country || null, region: region || null, source: source };
    if (region) setCookie('consent_region', region, COOKIE_DAYS);

    // Expose the resolved region to analytics: a GA4 user property (usable now —
    // gtag-only, no GTM) plus a dataLayer event (forward-compatible for a future GTM
    // container). `geo_region` = 'eu' (consent-required: EU+EEA+UK+CH+CA+BR) or 'row'.
    // Country/region are coarse (not PII) and the geo lookup is strictly necessary, so
    // this is not consent-gated. Fires only once the region is known (after the fetch).
    if (region) {
      try {
        window.dataLayer = window.dataLayer || [];
        window.dataLayer.push({ event: 'geo_resolved', geo_region: region, geo_country: country || '' });
        if (typeof window.gtag === 'function') {
          window.gtag('set', 'user_properties', { geo_region: region, geo_country: country || '' });
        }
      } catch (e) { /* non-critical */ }
    }

    try {
      document.dispatchEvent(new CustomEvent('geo:ready', { detail: window.__geo }));
    } catch (e) { /* CustomEvent unsupported — non-critical */ }
    applyBanner(region);
    if (window.console && console.debug) {
      console.debug('[geo-consent] country=' + (country || '(none)') +
        ' region=' + (region || '(none)') + ' source=' + source);
    }
  }

  function run() {
    // Keep the banner hidden until the region is known to require consent. This sets
    // html.cc-hide, which beats the theme's inline display:block via !important — so it
    // no longer matters whether the theme handler runs before or after us. A prior
    // choice is handled by applyBanner below.
    hideBanner();

    // 1) Test override: ?geo_country=XX (skips the network call).
    var override = new URLSearchParams(location.search).get('geo_country');
    if (override) { publish(override, regionFor(override), 'query-override'); return; }

    // 2) Edge fast path: the CloudFront viewer-response function sets a `geo_country`
    //    cookie (raw country code from CloudFront-Viewer-Country) on every response.
    //    When present it's authoritative and current (refreshed per request), so we
    //    derive the region from it and SKIP the network fetch entirely — this removes
    //    the geo round-trip even for first-time visitors. Region logic stays here only
    //    (single source); the edge stays dumb and just exposes the country.
    //    No-op until the edge function is deployed (see scripts/edge/geo-country-cookie.js).
    var edge = getCookie('geo_country');
    if (edge) { publish(edge, regionFor(edge), 'edge-cookie'); return; }

    // 3) Repeat-visit fast path: a cached EU region reveals the banner immediately
    //    (no wait for the fetch); row stays hidden. The fetch below still refreshes.
    var cached = getCookie('consent_region');
    if (cached) applyBanner(cached);

    // 4) Detect from the edge geo endpoint (fallback when no geo_country cookie).
    fetch(ENDPOINT, { credentials: 'omit' })
      .then(function (r) { return r.json(); })
      .then(function (g) { publish(g.country, regionFor(g.country), 'geo-endpoint'); })
      .catch(function () {
        // Fail-safe: unknown region → leave the banner shown (GDPR-safe default).
        publish(null, null, 'fetch-error');
      });
  }

  // Visibility is now a class on <html> (cc-hide/cc-show resolved by CSS), so our
  // decision wins regardless of when the theme handler runs — we no longer need to
  // fire last. Run as soon as the DOM is ready (the banner element need not exist
  // yet, since we toggle the <html> class, not the element).
  if (document.readyState !== 'loading') {
    run();
  } else {
    document.addEventListener('DOMContentLoaded', run);
  }
})();
