// YouTube Video Component - Dynamic Loading & Functionality
(function() {
    'use strict';

    // Auto-load CSS when this script loads
    loadYouTubeCSS();

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initYouTubeVideo);
    } else {
        initYouTubeVideo();
    }

    function loadYouTubeCSS() {
        if (!window.youtubeVideoCSSLoaded) {
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = '/css/youtube-video.css?v=' + (window.buildTimestamp || Date.now());
            document.head.appendChild(cssLink);
            window.youtubeVideoCSSLoaded = true;
        }
    }

    function initYouTubeVideo() {
        // Initialize lightbox
        initVideoLightbox();
        // Initialize lazy loading for video thumbnails  
        initVideoLazyLoading();
        // Mark as loaded
        window.youtubeVideoLoaded = true;
    }

    // Function to find the best available thumbnail quality
    function findBestThumbnail(img) {
        const videoId = img.getAttribute('data-video-id');
        if (!videoId) return;
        
        // List of thumbnail qualities to try, from highest to lowest
        const qualities = [
            { name: 'maxresdefault', minWidth: 1280, minHeight: 720 },
            { name: 'sddefault', minWidth: 640, minHeight: 480 },
            { name: 'hqdefault', minWidth: 480, minHeight: 360 },
            { name: 'mqdefault', minWidth: 320, minHeight: 180 }
        ];
        
        // Start with the highest quality and work down
        tryNextQuality(img, videoId, qualities, 0);
    }

    function tryNextQuality(img, videoId, qualities, index) {
        // If we've tried all qualities, stop
        if (index >= qualities.length) return;
        
        const quality = qualities[index];
        const testUrl = `https://img.youtube.com/vi/${videoId}/${quality.name}.jpg`;
        
        const testImg = new Image();
        
        testImg.onload = function() {
            // For maxresdefault, some videos return a 120x90 or 320x180 placeholder
            // Special handling for maxresdefault if dimensions are too small
            if (quality.name === 'maxresdefault' && (testImg.width < quality.minWidth || testImg.height < quality.minHeight)) {
                tryNextQuality(img, videoId, qualities, index + 1);
                return;
            }

            // YouTube sometimes returns valid thumbnails with slightly different dimensions than expected
            // Allow a margin of error (e.g., 90% of expected dimensions)
            const marginFactor = 0.9;
            const isAcceptableSize = testImg.width >= quality.minWidth * marginFactor &&
                                    testImg.height >= quality.minHeight * marginFactor;

            // Check if the image meets the minimum dimensions
            // This helps filter out placeholder images
            if (isAcceptableSize) {
                // This quality is good, use it
                img.src = testUrl;
            } else {
                // Try the next quality
                tryNextQuality(img, videoId, qualities, index + 1);
            }
        };
        
        testImg.onerror = function() {
            // Error loading this quality, try the next one
            tryNextQuality(img, videoId, qualities, index + 1);
        };
        
        // Start loading the test image
        testImg.src = testUrl;
    }

    function initVideoLazyLoading() {
        // Add click event listeners to all video thumbnails
        const videoThumbnails = document.querySelectorAll('.lazy-video-thumbnail');
        videoThumbnails.forEach(function(thumbnail) {
            thumbnail.addEventListener('click', function() {
                const container = thumbnail.closest('[data-video-id]');
                const videoId = container.getAttribute('data-video-id');
                const videoTitle = container.getAttribute('data-video-title');
                
                // Open video in lightbox modal
                openVideoInLightbox(videoId, videoTitle);
            });
        });

        // Initialize lazy loading for video thumbnails if not already handled
        if (typeof window.initLazyImages !== 'function') {
            window.initLazyImages = function() {
                const lazyImages = document.querySelectorAll('img.lazy-image');
                
                if ('IntersectionObserver' in window) {
                    const imageObserver = new IntersectionObserver(function(entries, observer) {
                        entries.forEach(function(entry) {
                            if (entry.isIntersecting) {
                                const image = entry.target;
                                if (image.dataset.src) {
                                    image.src = image.dataset.src;
                                    
                                    // For YouTube thumbnails, find the best quality after loading the initial image
                                    if (image.classList.contains('lazy-video-thumb-img') && image.dataset.videoId) {
                                        image.addEventListener('load', function() {
                                            findBestThumbnail(image);
                                        }, { once: true });
                                    }
                                }
                                
                                imageObserver.unobserve(image);
                            }
                        });
                    });
                    
                    lazyImages.forEach(function(image) {
                        imageObserver.observe(image);
                    });
                } else {
                    // Fallback for browsers that don't support IntersectionObserver
                    lazyImages.forEach(function(image) {
                        image.src = image.dataset.src || image.src;
                    });
                }
            };
            
            // Initialize lazy loading
            window.initLazyImages();
            
            // Re-check on window resize and orientation change
            window.addEventListener('resize', window.initLazyImages);
            window.addEventListener('orientationchange', window.initLazyImages);
        }
    }

    function initVideoLightbox() {
        // Create lightbox overlay if it doesn't exist
        if (!document.getElementById('video-lightbox-overlay')) {
            createLightboxOverlay();
        }
    }

    function createLightboxOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'video-lightbox-overlay';
        overlay.className = 'hidden fixed inset-0 bg-black bg-opacity-90 z-[9999] opacity-0 transition-opacity duration-300';
        
        overlay.innerHTML = `
            <button class="absolute top-4 right-4 bg-black bg-opacity-70 border-none text-white text-3xl cursor-pointer z-[10002] w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-opacity-90 hover:scale-110 pointer-events-auto" id="video-lightbox-close" aria-label="Close video">
                ×
            </button>
            <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[90%] max-w-6xl max-h-[90%] bg-black rounded-lg overflow-hidden shadow-[0_25px_50px_rgba(0,0,0,0.5)] pointer-events-auto">
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
        const closeBtn = overlay.querySelector('#video-lightbox-close');
        
        // Close button click
        closeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeLightbox();
        });
        
        // Click outside video container to close
        overlay.addEventListener('click', function(e) {
            // Only close if clicking on the overlay itself, not on the video container or close button
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

    function openVideoInLightbox(videoId, videoTitle) {
        const overlay = document.getElementById('video-lightbox-overlay');
        const iframe = document.getElementById('video-lightbox-iframe');
        
        if (overlay && iframe) {
            // Set video source
            iframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&modestbranding=1&iv_load_policy=3`;
            iframe.title = videoTitle || 'YouTube Video';
            iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
            
            // Show overlay with fade-in
            overlay.classList.remove('hidden');
            setTimeout(() => {
                overlay.classList.remove('opacity-0');
            }, 10);
            
            // Prevent body scroll
            document.body.style.overflow = 'hidden';
        }
    }

    function closeLightbox() {
        const overlay = document.getElementById('video-lightbox-overlay');
        const iframe = document.getElementById('video-lightbox-iframe');
        
        if (overlay && iframe) {
            // Fade out
            overlay.classList.add('opacity-0');
            
            // Hide after animation
            setTimeout(() => {
                overlay.classList.add('hidden');
                iframe.src = '';
            }, 300);
            
            // Restore body scroll
            document.body.style.overflow = '';
        }
    }

    // Export functions globally for use in components
    window.openVideoInLightbox = openVideoInLightbox;
    window.closeLightbox = closeLightbox;
    window.initVideoLightbox = initVideoLightbox;
    window.findBestThumbnail = findBestThumbnail;

})();
