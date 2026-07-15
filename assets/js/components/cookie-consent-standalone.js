// Standalone build of the theme cookie-consent banner JS for layouts that bypass
// the theme baseof and do not load the main.js bundle (e.g. design-system landing
// pages). Built by the sites' gulp component task to static/js/. Safe to include
// alongside main.js — cookie-consent.js carries an idempotency guard.
import '../cookie-consent.js';
