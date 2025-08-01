// Universal Video Handler
(function() {
    'use strict';

    // Create namespace for our module
    window.flowhuntMedia = window.flowhuntMedia || {};
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    function init() {
        // Create namespace for video handling
        window.flowhuntMedia = window.flowhuntMedia || {};
        window.flowhuntMedia.video = window.flowhuntMedia.video || {};
        
        // Expose initialization for external use
        window.flowhuntMedia.video.init = function() {
            // Check for any custom videos with lightbox that need initialization
            var customVideoContainers = document.querySelectorAll('[data-video-provider="custom"][data-video-lightbox="true"]');
            var needsInit = customVideoContainers.length > 0;
            
            // Force initialization if we have custom videos with lightbox
            if (needsInit) {
                initVideoTriggers();
                window.flowhuntMedia.video.handlersInitialized = true;
            }
            // Otherwise use standard initialization logic
            else if (window.flowhuntMedia && window.flowhuntMedia.videoLightbox) {
                // Only initialize video triggers if not already handled by video-lightbox.js
                if (!window.videoTriggersInitialized) {
                    initVideoTriggers();
                    window.videoTriggersInitialized = true;
                    window.flowhuntMedia.video.handlersInitialized = true;
                }
            } else {
                // Legacy mode - initialize everything
                initVideoTriggers();
                window.flowhuntMedia.video.handlersInitialized = true;
            }
            
            // Add GDPR event handlers if needed
            initGdprConsentHandlers();
        };
        
        // Initialize immediately
        window.flowhuntMedia.video.init();
    }

    function initVideoTriggers() {
        // YouTube videos
        var youtubeVideos = document.querySelectorAll('[data-video-provider="youtube"] .lazy-video-thumbnail');
        youtubeVideos.forEach(thumbnail => {
            thumbnail.addEventListener('click', handleYouTubeClick);
        });
        
        // Also add event listeners to play buttons
        var playButtons = document.querySelectorAll('[data-video-provider="youtube"] .lazy-video-play-button');
        playButtons.forEach(button => {
            button.addEventListener('click', handleYouTubeClick);
        });
        
        // Custom videos with lightbox capability
        var customVideos = document.querySelectorAll('[data-video-provider="custom"] video[data-video-lightbox="true"]');
        customVideos.forEach(video => {
            video.addEventListener('click', handleCustomVideoFullscreen);
            // Keep double-click for backward compatibility
            video.addEventListener('dblclick', handleCustomVideoFullscreen);
        });
        
        // Custom video containers with lightbox
        var customVideoContainers = document.querySelectorAll('[data-video-provider="custom"][data-video-lightbox="true"]');
        customVideoContainers.forEach(container => {
            // Add click handler to video element
            var video = container.querySelector('video');
            if (video) {
                video.addEventListener('click', function(e) {
                    if (e.target === video) {
                        handleCustomVideoFullscreen({ currentTarget: video });
                    }
                });
                
                // Keep double-click for backward compatibility
                video.addEventListener('dblclick', function(e) {
                    handleCustomVideoFullscreen({ currentTarget: video });
                });
            }
            
            // Add click to the container itself (as fallback)
            container.addEventListener('click', function(e) {
                // Only trigger if clicking on the container but not on the video controls
                if (e.target === container) {
                    var videoElement = container.querySelector('video');
                    if (videoElement) {
                        handleCustomVideoFullscreen({ currentTarget: videoElement });
                    }
                }
            });
        });
    }
    
    function handleYouTubeClick(e) {
        var target = e.currentTarget;
        var container;
        
        // Check if we clicked on the play button or the thumbnail
        if (target.classList.contains('lazy-video-play-button')) {
            // If clicked on play button, find the parent container
            container = target.closest('[data-video-id]');
        } else {
            // Otherwise assume we clicked on thumbnail
            container = target.closest('[data-video-id]');
        }
        
        if (!container) {
            console.error('Could not find video container');
            return;
        }
        
        var videoId = container.getAttribute('data-video-id');
        var videoTitle = container.getAttribute('data-video-title');
        var gdprCompliant = container.getAttribute('data-video-gdpr-compliant') === 'true';
        var autoplay = container.getAttribute('data-video-autoplay') === 'true';
        
        // Check for GDPR consent if required
        if (gdprCompliant && !hasGdprConsent()) {
            // Show GDPR consent overlay instead of playing video
            showGdprConsentOverlay(container);
            return;
        }
        
        // Open in lightbox
        if (window.flowhuntMedia && window.flowhuntMedia.videoLightbox) {
            // Create YouTube embed URL with all necessary parameters
            var youtubeUrl = gdprCompliant 
                ? `https://www.youtube-nocookie.com/embed/${videoId}` 
                : `https://www.youtube.com/embed/${videoId}`;
            
            // Add parameters needed for proper embedding
            youtubeUrl += '?rel=0&modestbranding=1&iv_load_policy=3&enablejsapi=1&origin=' + encodeURIComponent(window.location.origin);
            
            // Add required YouTube parameters to prevent "Video unavailable" errors
            youtubeUrl += '&wmode=transparent&widget_referrer=' + encodeURIComponent(window.location.href);
            
            // Always enable autoplay when clicking on video thumbnail or play button
            youtubeUrl += '&autoplay=1'; // Without mute parameter to enable sound
            
            window.flowhuntMedia.videoLightbox.open({
                src: youtubeUrl,
                title: videoTitle,
                provider: 'youtube',
                autoplay: true, // Always set to true for lightbox
                gdprCompliant: gdprCompliant
            });
        } else if (window.openVideoInLightbox) {
            // Fallback to legacy function
            window.openVideoInLightbox(videoId, videoTitle);
        }
    }
    
    function handleCustomVideoFullscreen(e) {
        if (e.preventDefault) {
            e.preventDefault();
        }
        
        var video, container;
        
        if (e.currentTarget && e.currentTarget.tagName === 'VIDEO') {
            video = e.currentTarget;
            container = video.closest('[data-video-provider="custom"]');
        } else if (e.currentTarget) {
            container = e.currentTarget;
            video = container.querySelector('video');
        }
        
        if (!video && container) {
            video = container.querySelector('video');
        }
        
        if (!container && !video) {
            console.error("Could not find video container or video element");
            return;
        }
        
        // Get video data - prioritize data attributes on container over video element
        var src = '';
        if (container) {
            src = container.getAttribute('data-video-src') || '';
        }
        if (!src && video) {
            src = video.getAttribute('data-video-src') || '';
        }
        if (!src && video && video.querySelector('source')) {
            src = video.querySelector('source').src || '';
        }
                  
        var title = '';
        if (container) {
            title = container.getAttribute('data-video-title') || '';
        }
        if (!title && video) {
            title = video.getAttribute('title') || '';
        }
        if (!title) {
            title = 'Video';
        }
                   
        var poster = '';
        if (video) {
            poster = video.poster || '';
        }
        
        // Open in lightbox
        if (window.flowhuntMedia && window.flowhuntMedia.videoLightbox && src) {
            try {
                window.flowhuntMedia.videoLightbox.open({
                    src: src,
                    title: title,
                    provider: 'custom',
                    autoplay: true,
                    poster: poster
                });
            } catch (error) {
                console.error('Error opening video in lightbox:', error);
                // Fallback if lightbox component throws an error
                window.open(src, '_blank');
            }
        } else if (window.openVideoInLightbox && src) {
            // Try legacy function
            try {
                window.openVideoInLightbox(src, title, 'custom', true, poster);
            } catch (error) {
                console.error('Error opening video in legacy lightbox:', error);
                // Fallback if legacy lightbox component throws an error
                window.open(src, '_blank');
            }
        } else if (src) {
            // Fallback if lightbox component is not available
            window.open(src, '_blank');
        } else {
            console.error('No source found for video');
        }
    }
    
    // GDPR Consent Handling
    function hasGdprConsent() {
        // Check for video-specific consent first
        var videoConsent = localStorage.getItem('video-consent') === 'granted' || 
                          document.cookie.indexOf('video-consent=granted') > -1;
        
        // If we already have video-specific consent, return true
        if (videoConsent) {
            return true;
        }
        
        // Check for global cookie consent as fallback
        // If user accepted all cookies (analytics), we can assume YouTube consent
        var globalConsent = false;
        
        // Helper function to get cookie value
        function getCookie(name) {
            var nameEQ = name + "=";
            var ca = document.cookie.split(';');
            for(var i=0; i < ca.length; i++) {
                var c = ca[i].trim();
                if (c.indexOf(nameEQ) === 0) {
                    return c.substring(nameEQ.length, c.length);
                }
            }
            return null;
        }
        
        // Check global cookie consent status
        var cookieConsentStatus = getCookie('cookie_consent_status');
        globalConsent = cookieConsentStatus === 'all';
        
        console.log("Checking GDPR consent...");
        console.log("Video consent:", videoConsent);
        console.log("Global cookie consent:", cookieConsentStatus);
        
        return videoConsent || globalConsent;
    }
    
    function setGdprConsent(granted) {
        if (granted) {
            localStorage.setItem('video-consent', 'granted');
            document.cookie = 'video-consent=granted; path=/; max-age=31536000'; // 1 year
            
            // Try to sync with global cookie consent if possible
            // But only do this if we don't already have a global cookie consent
            var globalConsent = document.cookie.indexOf('cookie_consent_status=') > -1;
            if (!globalConsent && typeof CookieManager !== 'undefined' && CookieManager.set) {
                CookieManager.set('cookie_consent_status', 'all', 365);
                
                // If we have gtag, update consent there too
                if (typeof window.gtag === 'function') {
                    window.gtag('consent', 'update', {
                        'analytics_storage': 'granted',
                        'ad_storage': 'granted',
                        'ad_user_data': 'granted',
                        'ad_personalization': 'granted'
                    });
                }
            }
        } else {
            localStorage.removeItem('video-consent');
            document.cookie = 'video-consent=; path=/; max-age=0'; // Remove cookie
            
            // We don't revoke global cookie consent as it's a separate concern
            // and user might want analytics but not YouTube
        }
    }
    
    function showGdprConsentOverlay(container) {
        const overlay = container.querySelector('.gdpr-consent-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.classList.add('flex', 'items-center', 'justify-center');
        }
    }
    
    function hideGdprConsentOverlay(container) {
        const overlay = container.querySelector('.gdpr-consent-overlay');
        if (overlay) {
            overlay.style.display = 'none';
            overlay.classList.remove('flex', 'items-center', 'justify-center');
        }
    }
    
    function initGdprConsentHandlers() {
        // Accept buttons
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList.contains('gdpr-accept-btn')) {
                e.preventDefault();
                
                // Get container and video data
                const container = e.target.closest('[data-video-id]');
                if (container) {
                    // Set consent
                    setGdprConsent(true);
                    
                    // Hide overlay
                    hideGdprConsentOverlay(container);
                    
                    // Trigger video click
                    const thumbnail = container.querySelector('.lazy-video-thumbnail');
                    if (thumbnail) {
                        thumbnail.click();
                    }
                }
            }
        });
        
        // Decline buttons
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList.contains('gdpr-decline-btn')) {
                e.preventDefault();
                
                // Get container
                var container = e.target.closest('[data-video-id]');
                if (container) {
                    // Hide overlay without setting consent
                    hideGdprConsentOverlay(container);
                }
            }
        });
    }
    
    // Export functions to namespace
    window.flowhuntMedia.video = {
        init: init,
        initTriggers: initVideoTriggers,
        hasGdprConsent: hasGdprConsent,
        setGdprConsent: setGdprConsent,
        // Add a manual trigger function
        openCustomVideo: function(containerSelector) {
            var container = document.querySelector(containerSelector);
            if (container) {
                var video = container.querySelector('video');
                if (video) {
                    handleCustomVideoFullscreen({ currentTarget: video });
                    return true;
                }
            }
            return false;
        }
    };
    
    // Add error handling for YouTube videos
    window.addEventListener('message', function(event) {
        // Check if message is from YouTube
        if (typeof event.data === 'string' && event.data.indexOf('{"event":"error"') === 0) {
            try {
                var data = JSON.parse(event.data);
                if (data.event === 'error' && data.info && data.id) {
                    console.warn('YouTube embed error:', data.info);
                    
                    // If we're in the lightbox, show a more user-friendly error
                    var lightbox = document.getElementById('video-lightbox-overlay');
                    if (lightbox && !lightbox.classList.contains('hidden')) {
                        var iframe = document.getElementById('video-lightbox-iframe');
                        if (iframe && iframe.id === data.id) {
                            // Show a friendly error message
                            showVideoError(data.info);
                        }
                    }
                }
            } catch (e) {
                // Not a JSON message or parsing error
            }
        }
    });
    
    function showVideoError(errorInfo) {
        var container = document.querySelector('#video-lightbox-overlay .video-aspect-ratio');
        if (container) {
            // Try to get fallback URL from the original video element that triggered the lightbox
            var fallbackUrl = '';
            var videoElements = document.querySelectorAll('[data-video-id]');
            for (var i = 0; i < videoElements.length; i++) {
                if (videoElements[i].getAttribute('data-video-fallback-url')) {
                    fallbackUrl = videoElements[i].getAttribute('data-video-fallback-url');
                    break;
                }
            }
            
            // Create error message
            var errorDiv = document.createElement('div');
            errorDiv.className = 'video-error-message bg-black text-white p-4 flex flex-col items-center justify-center h-full';
            
            var errorHtml = `
                <div class="text-center">
                    <h3 class="text-xl font-bold mb-2">Video Unavailable</h3>
                    <p class="mb-4">Sorry, this video cannot be played in embedded mode.</p>
            `;
            
            if (fallbackUrl) {
                errorHtml += `
                    <div class="flex flex-col sm:flex-row gap-3 justify-center">
                        <a href="${fallbackUrl}" target="_blank" rel="noopener noreferrer" class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded">
                            Watch on YouTube
                        </a>
                        <button class="bg-transparent hover:bg-gray-700 text-white font-semibold py-2 px-4 border border-white hover:border-transparent rounded close-video-btn">
                            Close
                        </button>
                    </div>
                `;
            } else {
                errorHtml += `
                    <button class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded close-video-btn">
                        Close
                    </button>
                `;
            }
            
            errorHtml += `</div>`;
            errorDiv.innerHTML = errorHtml;
            
            // Clear container and add error message
            container.innerHTML = '';
            container.appendChild(errorDiv);
            
            // Add event listener to close button
            var closeBtn = errorDiv.querySelector('.close-video-btn');
            if (closeBtn) {
                closeBtn.addEventListener('click', function() {
                    if (window.closeLightbox) {
                        window.closeLightbox();
                    }
                });
            }
        }
    }
    
    // Backward compatibility
    window.initVideoHandlers = init;
    window.hasVideoGdprConsent = hasGdprConsent;
    window.setVideoGdprConsent = setGdprConsent;
})();
