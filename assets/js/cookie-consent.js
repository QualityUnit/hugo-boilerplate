/**
 * Cookie Consent - Zjednodušená implementácia
 */

// Jednoduché funkcie pre prácu s cookies
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

// Hlavná funkcionalita pre cookie consent
document.addEventListener('DOMContentLoaded', function() {
  // Elementy
  const banner = document.getElementById('cookie-consent-banner');
  const modal = document.getElementById('cookie-settings-modal');
  const analyticsCheckbox = document.getElementById('analytics-cookies');

  // Stav cookies
  const consentStatus = CookieManager.get('cookie_consent_status');

  // Odstránenie inline štýlov, ktoré by mohli prekážať
  if (banner) {
    banner.removeAttribute('style');
  }

  // Skryť/zobraziť banner podľa stavu cookies
  if (consentStatus) {
    hideBanner();
  } else {
    showBanner();
  }

  // Aplikovať predchádzajúci výber
  if (consentStatus === 'all' && analyticsCheckbox) {
    analyticsCheckbox.checked = true;
    updateAnalyticsConsent(true);
  }

  // Event handlery pre tlačidlá
  document.addEventListener('click', function(event) {
    // Akceptovať všetky cookies
    if (event.target.closest('[data-cookie-consent="accept-all"]')) {
      setConsent('all');
      hideBanner();
    }

    // Akceptovať len nevyhnutné cookies
    if (event.target.closest('[data-cookie-consent="accept-necessary"]')) {
      setConsent('necessary');
      hideBanner();
    }

    // Otvoriť nastavenia
    if (event.target.closest('[data-cookie-consent="settings"]')) {
      modal?.classList.remove('hidden');
    }

    // Zatvoriť nastavenia
    if (event.target.closest('[data-cookie-settings-close]')) {
      modal?.classList.add('hidden');
    }

    // Uložiť nastavenia
    if (event.target.closest('[data-cookie-settings-save]')) {
      const allowAnalytics = analyticsCheckbox?.checked || false;
      setConsent(allowAnalytics ? 'all' : 'necessary');
      modal?.classList.add('hidden');
      hideBanner();
    }
  });

  // Funkcia na skrytie bannera
  function hideBanner() {
    if (banner) {
      // Odstránenie akýchkoľvek inline štýlov
      banner.removeAttribute('style');
      // Pridanie triedy pre skrytie
      banner.classList.add('cookie-hidden');
      // Dodatočné nastavenie pre prípad, že CSS zlyhá
      banner.style.display = 'none';
      banner.style.visibility = 'hidden';
      banner.style.opacity = '0';
    }
  }

  // Funkcia na zobrazenie bannera
  function showBanner() {
    if (banner) {
      banner.classList.remove('cookie-hidden');
      banner.style.display = 'block';
      banner.style.visibility = 'visible';
      banner.style.opacity = '1';
    }
  }

  // Nastaviť cookies a aplikovať nastavenia
  function setConsent(level) {
    CookieManager.set('cookie_consent_status', level, 365);
    updateAnalyticsConsent(level === 'all');
    console.log('Cookie consent set to:', level); // Pre debugovanie
  }

  // Aktualizovať nastavenia pre Google Analytics
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
