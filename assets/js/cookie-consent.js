/**
 * Cookie Consent Script
 */

const CookieManager = {
  set(name, value, days) {
    const date = new Date();
    date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
    document.cookie = `${name}=${value};expires=${date.toUTCString()};path=/`;
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
      setConsent('all');
      hideBanner();
    }

    if (event.target.closest('[data-cookie-consent="accept-necessary"]')) {
      setConsent('necessary');
      hideBanner();
    }

    if (event.target.closest('[data-cookie-consent="settings"]')) {
      modal?.classList.remove('hidden');
    }

    if (event.target.closest('[data-cookie-settings-close]')) {
      modal?.classList.add('hidden');
    }

    if (event.target.closest('[data-cookie-settings-save]')) {
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
    if (typeof window.gtag === 'function') {
      window.gtag('consent', 'update', {
        'analytics_storage': allowed ? 'granted' : 'denied',
        'ad_storage': allowed ? 'granted' : 'denied',
        'ad_user_data': allowed ? 'granted' : 'denied',
        'ad_personalization': allowed ? 'granted' : 'denied'
      });
    }
  }
});
