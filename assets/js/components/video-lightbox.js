// Universal Video Lightbox Component
(function() {
    'use strict';
    
    // Add support for GDPR consent mode - defining this at the top to avoid temporal dead zone issues
    var gdprConsentMode = false;

    // Auto-load CSS when this script loads
    loadLightboxCSS();

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initVideoLightbox);
    } else {
        initVideoLightbox();
    }

    function loadLightboxCSS() {
        if (!window.videoLightboxCSSLoaded) {
            var cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = '/css/video-lightbox.css?v=' + (window.buildTimestamp || Date.now());
            document.head.appendChild(cssLink);
            window.videoLightboxCSSLoaded = true;
        }
    }

    // Check for GDPR consent cookie or localStorage setting
    function checkGdprConsent() {
        // Check for consent in localStorage or cookies
        return localStorage.getItem('video-consent') === 'granted' || 
               document.cookie.indexOf('video-consent=granted') > -1;
    }

    // Method to set GDPR consent mode for all videos
    function setGdprConsentMode(enabled) {
        gdprConsentMode = !!enabled;
        
        // Store consent in localStorage and cookie for persistence
        if (gdprConsentMode) {
            localStorage.setItem('video-consent', 'granted');
            document.cookie = 'video-consent=granted; path=/; max-age=31536000'; // 1 year
        }
    }

    function initVideoLightbox() {
        // Create lightbox overlay if it doesn't exist
        if (!document.getElementById('video-lightbox-overlay')) {
            createLightboxOverlay();
        }

        // Check if GDPR consent is required
        gdprConsentMode = checkGdprConsent();

        // Initialize all video triggers with data attributes
        initVideoTriggers();
        
        // Mark that triggers have been initialized
        window.videoTriggersInitialized = true;
    }

    function initVideoTriggers() {
        // Skip if handlers are already handled by video-handler.js
        if (window.flowhuntMedia && window.flowhuntMedia.video && window.flowhuntMedia.video.handlersInitialized) {
            return;
        }
        
        // Find all direct video trigger elements
        var videoTriggers = document.querySelectorAll('[data-video-url], [data-video-embed]');
        
        videoTriggers.forEach(function(trigger) {
            trigger.removeEventListener('click', handleVideoTriggerClick);
            trigger.addEventListener('click', handleVideoTriggerClick);
        });
        
        // Find all YouTube containers
        var videoContainers = document.querySelectorAll('[data-video-id]');
        videoContainers.forEach(function(container) {
            // Look for a specific trigger inside
            var trigger = container.querySelector('[data-video-trigger="true"]');
            if (trigger) {
                trigger.removeEventListener('click', handleVideoTriggerClick);
                trigger.addEventListener('click', handleVideoTriggerClick);
            } else {
                // No specific trigger found, use the thumbnail
                var thumbnail = container.querySelector('.lazy-video-thumbnail');
                if (thumbnail) {
                    thumbnail.removeEventListener('click', handleVideoTriggerClick);
                    thumbnail.addEventListener('click', handleVideoTriggerClick);
                }
            }
            
            // Also add event listener to play button if present
            var playButton = container.querySelector('.lazy-video-play-button');
            if (playButton) {
                playButton.removeEventListener('click', handleVideoTriggerClick);
                playButton.addEventListener('click', handleVideoTriggerClick);
            }
        });
    }

    function handleVideoTriggerClick(event) {
        event.preventDefault();
        event.stopPropagation();
        
        var trigger = event.currentTarget;
        var container = trigger;
        
        // If this is a thumbnail or play button, find the parent container
        if (trigger.classList.contains('lazy-video-thumbnail') || trigger.classList.contains('lazy-video-play-button')) {
            container = trigger.closest('[data-video-id]');
        }
        
        // Get video data from the container or trigger
        var videoId = container.getAttribute('data-video-id');
        var videoUrl = container.getAttribute('data-video-url');
        var videoEmbed = container.getAttribute('data-video-embed');
        var videoTitle = container.getAttribute('data-video-title') || 'Video';
        var autoplay = container.getAttribute('data-video-autoplay') === 'true';
        var gdprCompliant = container.getAttribute('data-video-gdpr-compliant') === 'true';
        var provider = container.getAttribute('data-video-provider') || 'custom';
        
        var videoData = {
            src: null,
            title: videoTitle,
            provider: provider,
            autoplay: autoplay,
            width: container.getAttribute('data-video-width'),
            height: container.getAttribute('data-video-height'),
            gdprCompliant: gdprCompliant
        };

        // Priority order: direct embed URL, then YouTube ID, then raw URL
        if (videoEmbed) {
            videoData.src = videoEmbed;
        } else if (videoId) {
            // Handle GDPR mode for certain providers
            var shouldApplyGdprParams = (gdprConsentMode || gdprCompliant) && 
                                          (provider === 'youtube' || provider === 'vimeo');
            
            switch (provider) {
                case 'youtube':
                    // Always enable autoplay for YouTube videos in lightbox - with sound since we have user interaction
                    videoData.src = `https://www.youtube.com/embed/${videoId}?rel=0&modestbranding=1&iv_load_policy=3&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}&wmode=transparent&widget_referrer=${encodeURIComponent(window.location.href)}&autoplay=1`;
                    
                    // Add YouTube privacy-enhanced mode if GDPR compliance is required
                    if (shouldApplyGdprParams) {
                        videoData.src = `https://www.youtube-nocookie.com/embed/${videoId}?rel=0&modestbranding=1&iv_load_policy=3&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}&wmode=transparent&widget_referrer=${encodeURIComponent(window.location.href)}&autoplay=1`;
                    }
                    
                    // Force autoplay flag to be true
                    videoData.autoplay = true;
                    break;
                case 'vimeo':
                    videoData.src = `https://player.vimeo.com/video/${videoId}?title=0&byline=0&portrait=0${autoplay ? '&autoplay=1' : ''}`;
                    
                    // Add Vimeo's Do Not Track parameter if GDPR compliance is required
                    if (shouldApplyGdprParams) {
                        videoData.src += '&dnt=1';
                    }
                    break;
                default:
                    console.warn('Unknown provider:', provider);
                    break;
            }
        } else if (videoUrl) {
            videoData.src = videoUrl;
        }

        if (videoData.src) {
            openVideoInLightbox(videoData);
        } else {
            console.error('No valid video source found in data attributes');
        }
    }

    function extractVideoData(element) {
        if (!element) {
            console.error('No element provided to extractVideoData');
            return { src: null };
        }
        
        var data = {
            src: null,
            title: element.getAttribute('data-video-title') || 'Video',
            provider: element.getAttribute('data-video-provider') || 'custom',
            autoplay: element.getAttribute('data-video-autoplay') === 'true',
            width: element.getAttribute('data-video-width'),
            height: element.getAttribute('data-video-height'),
            gdprCompliant: element.getAttribute('data-video-gdpr-compliant') === 'true'
        };

        // Priority order: direct embed URL, then YouTube ID, then raw URL
        if (element.getAttribute('data-video-embed')) {
            data.src = element.getAttribute('data-video-embed');
        } else if (element.getAttribute('data-video-id')) {
            var videoId = element.getAttribute('data-video-id');
            var provider = data.provider.toLowerCase();
            
            // Handle GDPR mode for certain providers
            var shouldApplyGdprParams = (gdprConsentMode || data.gdprCompliant) && 
                                          (provider === 'youtube' || provider === 'vimeo');
            
            switch (provider) {
                case 'youtube':
                    // Always enable autoplay for YouTube videos in lightbox - with sound since we have user interaction
                    data.src = `https://www.youtube.com/embed/${videoId}?rel=0&modestbranding=1&iv_load_policy=3&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}&wmode=transparent&widget_referrer=${encodeURIComponent(window.location.href)}&autoplay=1`;
                    
                    // Add YouTube privacy-enhanced mode if GDPR compliance is required
                    if (shouldApplyGdprParams) {
                        data.src = `https://www.youtube-nocookie.com/embed/${videoId}?rel=0&modestbranding=1&iv_load_policy=3&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}&wmode=transparent&widget_referrer=${encodeURIComponent(window.location.href)}&autoplay=1`;
                    }
                    
                    // Force autoplay flag to be true
                    data.autoplay = true;
                    break;
                case 'vimeo':
                    data.src = `https://player.vimeo.com/video/${videoId}?title=0&byline=0&portrait=0${data.autoplay ? '&autoplay=1' : ''}`;
                    
                    // Add Vimeo's Do Not Track parameter if GDPR compliance is required
                    if (shouldApplyGdprParams) {
                        data.src += '&dnt=1';
                    }
                    break;
                default:
                    console.warn('Unknown provider:', provider);
                    break;
            }
        } else if (element.getAttribute('data-video-url')) {
            data.src = element.getAttribute('data-video-url');
        }

        return data;
    }

    function createLightboxOverlay() {
        var overlay = document.createElement('div');
        overlay.id = 'video-lightbox-overlay';
        overlay.className = 'hidden fixed inset-0 bg-black bg-opacity-90 z-[9999] opacity-0 transition-opacity duration-300';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-label', 'Video player');
        
        overlay.innerHTML = `
            <button class="fixed top-4 right-4 bg-black bg-opacity-70 border-none text-white text-3xl cursor-pointer z-[10002] w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-opacity-90 hover:scale-110" id="video-lightbox-close" aria-label="Close video">
                ×
            </button>
            <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[90%] max-w-6xl max-h-[90%] bg-black rounded-lg overflow-hidden shadow-[0_25px_50px_rgba(0,0,0,0.5)]">
                <div class="video-aspect-ratio bg-black">
                    <iframe id="video-lightbox-iframe" class="w-full h-full border-none" allowfullscreen></iframe>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        // Add event listeners
        setupLightboxEventListeners(overlay);
    }

    function setupLightboxEventListeners(overlay) {
        var closeBtn = overlay.querySelector('#video-lightbox-close');
        
        // Close button click
        closeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeLightbox();
        });
        
        // Click outside video container to close
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                closeLightbox();
            }
        });
        
        // Escape key to close
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && !overlay.classList.contains('hidden')) {
                closeLightbox();
            }
        });
    }

function openVideoInLightbox(videoData) {
        var overlay = document.getElementById('video-lightbox-overlay');
        var container = overlay.querySelector('.video-aspect-ratio');
        
        if (overlay && container && videoData.src) {
            // Clear container first
            container.innerHTML = '';
            
            // Check if this is a direct video file URL (mp4, webm, etc.)
            var isVideoFile = typeof videoData.src === 'string' && 
                               (videoData.src.match(/\.(mp4|webm|ogg|mov)($|\?)/i));
            
            if (isVideoFile) {
                // Create HTML5 video element for direct video files
                var video = document.createElement('video');
                video.id = 'video-lightbox-player';
                video.className = 'w-full h-full border-none';
                video.controls = true;
                video.autoplay = videoData.autoplay;
                video.preload = 'metadata';
                video.title = videoData.title;
                
                // Add poster if available
                if (videoData.poster) {
                    video.poster = videoData.poster;
                }
                
                // Add source
                var source = document.createElement('source');
                source.src = videoData.src;
                source.type = getVideoMimeType(videoData.src);
                video.appendChild(source);
                
                // Add fallback text
                video.innerHTML += 'Your browser does not support the video tag.';
                
                // Append to container
                container.appendChild(video);
            } else {
                // Use iframe for embeddable videos (YouTube, Vimeo, etc.)
                var iframe = document.createElement('iframe');
                iframe.id = 'video-lightbox-iframe';
                iframe.className = 'w-full h-full border-none';
                iframe.src = videoData.src;
                iframe.title = videoData.title;
                
                // Enhanced allow attributes for better video playback
                iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
                iframe.allowFullscreen = true;
                iframe.setAttribute('frameborder', '0');
                
                // For YouTube videos, add extra attributes to help prevent "Video unavailable" errors
                if (videoData.provider === 'youtube') {
                    iframe.setAttribute('allowtransparency', 'true');
                    iframe.setAttribute('allowscriptaccess', 'always');
                    iframe.referrerPolicy = 'origin';
                    
                    // Ensure autoplay parameter is added for YouTube
                    if (!videoData.src.includes('autoplay=')) {
                        videoData.src += '&autoplay=1'; // Remove mute parameter to enable sound
                    }
                    
                    // Add wmode parameter if not already in the URL
                    if (!videoData.src.includes('wmode=')) {
                        videoData.src += '&wmode=transparent';
                    }
                    
                    // Add widget_referrer parameter if not already in the URL
                    if (!videoData.src.includes('widget_referrer=')) {
                        videoData.src += `&widget_referrer=${encodeURIComponent(window.location.href)}`;
                    }
                    
                    // Add event listeners for YouTube iframe
                    iframe.addEventListener('load', function() {
                        // Mark as loaded successfully
                        iframe.dataset.loaded = 'true';
                    });
                    
                    iframe.addEventListener('error', function() {
                        // If iframe errors, try alternate embedding method
                        console.warn('YouTube iframe load error, trying alternate method');
                        retryYouTubeEmbed(container, videoData);
                    });
                    
                    // Add a timeout to check if the iframe loaded correctly
                    setTimeout(function() {
                        if (iframe.dataset.loaded !== 'true') {
                            console.warn('YouTube iframe load timeout, trying alternate method');
                            retryYouTubeEmbed(container, videoData);
                        }
                    }, 5000); // 5-second timeout
                    
                    // Update the iframe src
                    iframe.src = videoData.src;
                }
                
                if (videoData.width) iframe.width = videoData.width;
                if (videoData.height) iframe.height = videoData.height;
                
                // Append to container
                container.appendChild(iframe);
            }
            
            // Show overlay with fade-in
            overlay.classList.remove('hidden');
            setTimeout(() => {
                overlay.classList.remove('opacity-0');
            }, 10);
            
            // Prevent body scroll
            document.body.style.overflow = 'hidden';
        }
    }
    
    // Helper function to get MIME type from file extension
    function getVideoMimeType(url) {
        if (url.match(/\.mp4($|\?)/i)) return 'video/mp4';
        if (url.match(/\.webm($|\?)/i)) return 'video/webm';
        if (url.match(/\.ogg($|\?)/i)) return 'video/ogg';
        if (url.match(/\.mov($|\?)/i)) return 'video/quicktime';
        return 'video/mp4'; // Default
    }

    function closeLightbox() {
        var overlay = document.getElementById('video-lightbox-overlay');
        var container = overlay.querySelector('.video-aspect-ratio');
        
        if (overlay && container) {
            // Fade out
            overlay.classList.add('opacity-0');
            
            // Hide after animation
            setTimeout(() => {
                overlay.classList.add('hidden');
                
                // Clean up container - removing iframe or video to stop playback
                container.innerHTML = '';
                
                // Create a new empty iframe as a placeholder
                var iframe = document.createElement('iframe');
                iframe.id = 'video-lightbox-iframe';
                iframe.className = 'w-full h-full border-none';
                container.appendChild(iframe);
            }, 300);
            
            // Restore body scroll
            document.body.style.overflow = '';
        }
    }
    
    // Function to retry YouTube embed with alternate method if the standard embed fails
    function retryYouTubeEmbed(container, videoData) {
        // Only proceed if we have a valid container and it's for YouTube
        if (!container || videoData.provider !== 'youtube') return;
        
        // Extract video ID from src
        var videoId = '';
        var srcUrl = videoData.src;
        
        if (srcUrl.includes('youtube.com/embed/') || srcUrl.includes('youtube-nocookie.com/embed/')) {
            var matches = srcUrl.match(/\/embed\/([a-zA-Z0-9_-]+)/);
            if (matches && matches[1]) {
                videoId = matches[1];
            }
        }
        
        if (!videoId) {
            console.error('Could not extract YouTube video ID for retry');
            return;
        }
        
        // Try to find a fallback URL from the original video element
        var fallbackUrl = '';
        var videoElements = document.querySelectorAll('[data-video-id="' + videoId + '"]');
        for (var i = 0; i < videoElements.length; i++) {
            if (videoElements[i].getAttribute('data-video-fallback-url')) {
                fallbackUrl = videoElements[i].getAttribute('data-video-fallback-url');
                break;
            }
        }
        
        // Create fallback HTML
        var fallbackHtml = `
            <div class="video-error-message bg-black text-white p-4 flex flex-col items-center justify-center h-full">
                <div class="text-center">
                    <h3 class="text-xl font-bold mb-2">Video Embedding Issue</h3>
                    <p class="mb-4">This video cannot be embedded directly in the lightbox.</p>
                    ${fallbackUrl ? `
                        <div class="flex flex-col sm:flex-row gap-3 justify-center">
                            <a href="${fallbackUrl}" target="_blank" rel="noopener noreferrer" class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded">
                                Watch on YouTube
                            </a>
                            <button class="bg-transparent hover:bg-gray-700 text-white font-semibold py-2 px-4 border border-white hover:border-transparent rounded close-video-btn">
                                Close
                            </button>
                        </div>
                    ` : `
                        <button class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded close-video-btn">
                            Close
                        </button>
                    `}
                </div>
            </div>
        `;
        
        // Clear container and add fallback message
        container.innerHTML = fallbackHtml;
        
        // Add event listener to close button
        var closeBtn = container.querySelector('.close-video-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                closeLightbox();
            });
        }
    }

    // Export functions globally
    window.openVideoInLightbox = openVideoInLightbox;
    window.closeLightbox = closeLightbox;
    window.initVideoLightbox = initVideoLightbox;
    window.initVideoTriggers = initVideoTriggers;
    window.setVideoGdprConsent = setGdprConsentMode;

    // Create a namespace for better organization
    window.flowhuntMedia = window.flowhuntMedia || {};
    window.flowhuntMedia.videoLightbox = {
        open: openVideoInLightbox,
        close: closeLightbox,
        init: initVideoLightbox,
        initTriggers: initVideoTriggers,
        setGdprConsent: setGdprConsentMode
    };

    // Backward compatibility
    window.videoLightbox = {
        open: openVideoInLightbox,
        close: closeLightbox,
        init: initVideoLightbox,
        initTriggers: initVideoTriggers,
        setGdprConsent: setGdprConsentMode
    };

})();
