// Lazy loading implementation for images, videos, and SVGs
document.addEventListener('DOMContentLoaded', function() {
  // Initialize lazy loading for images
  function initLazyImages() {
    const lazyImages = document.querySelectorAll('img.lazy-image');
    
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver(function(entries, observer) {
        entries.forEach(function(entry) {
          if (entry.isIntersecting) {
            const image = entry.target;
            if (image.dataset.src) {
              image.src = image.dataset.src;
              
              // Add onload event to add the 'loaded' class
              image.onload = function() {
                image.classList.add('loaded');
              };
              
              // For YouTube thumbnails, find the best quality after loading the initial image
              if (image.classList.contains('lazy-video-thumb-img') && image.dataset.videoId) {
                image.addEventListener('load', function() {
                  // This function would be defined in lazyvideo.html
                  if (typeof findBestThumbnail === 'function') {
                    findBestThumbnail(image);
                  }
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
      let lazyImageTimeout;
      
      function lazyLoadImages() {
        if (lazyImageTimeout) {
          clearTimeout(lazyImageTimeout);
        }
        
        lazyImageTimeout = setTimeout(function() {
          const scrollTop = window.pageYOffset;
          
          lazyImages.forEach(function(img) {
            if (img.offsetTop < (window.innerHeight + scrollTop)) {
              if (img.dataset.src) {
                img.src = img.dataset.src;
                
                // Add onload event to add the 'loaded' class
                img.onload = function() {
                  img.classList.add('loaded');
                };
              }
            }
          });
          
          if (lazyImages.length === 0) {
            document.removeEventListener('scroll', lazyLoadImages);
            window.removeEventListener('resize', lazyLoadImages);
            window.removeEventListener('orientationChange', lazyLoadImages);
          }
        }, 20);
      }
      
      document.addEventListener('scroll', lazyLoadImages);
      window.addEventListener('resize', lazyLoadImages);
      window.addEventListener('orientationChange', lazyLoadImages);
      
      // Initial load
      lazyLoadImages();
    }
  }
  
  // Initialize lazy loading for SVGs
  function initLazySVGs() {
    const lazySVGs = document.querySelectorAll('object.lazy-svg');
    
    if ('IntersectionObserver' in window) {
      const svgObserver = new IntersectionObserver(function(entries, observer) {
        entries.forEach(function(entry) {
          if (entry.isIntersecting) {
            const svg = entry.target;
            if (svg.dataset.src) {
              svg.data = svg.dataset.src;
              
              // Add onload event to add the 'loaded' class
              svg.onload = function() {
                svg.classList.add('loaded');
              };
            }
            
            svgObserver.unobserve(svg);
          }
        });
      });
      
      lazySVGs.forEach(function(svg) {
        svgObserver.observe(svg);
      });
    } else {
      // Fallback for browsers that don't support IntersectionObserver
      let lazySVGTimeout;
      
      function lazyLoadSVGs() {
        if (lazySVGTimeout) {
          clearTimeout(lazySVGTimeout);
        }
        
        lazySVGTimeout = setTimeout(function() {
          const scrollTop = window.pageYOffset;
          
          lazySVGs.forEach(function(svg) {
            if (svg.offsetTop < (window.innerHeight + scrollTop)) {
              if (svg.dataset.src) {
                svg.data = svg.dataset.src;
                
                // Add onload event to add the 'loaded' class
                svg.onload = function() {
                  svg.classList.add('loaded');
                };
              }
            }
          });
          
          if (lazySVGs.length === 0) {
            document.removeEventListener('scroll', lazyLoadSVGs);
            window.removeEventListener('resize', lazyLoadSVGs);
            window.removeEventListener('orientationChange', lazyLoadSVGs);
          }
        }, 20);
      }
      
      document.addEventListener('scroll', lazyLoadSVGs);
      window.addEventListener('resize', lazyLoadSVGs);
      window.addEventListener('orientationChange', lazyLoadSVGs);
      
      // Initial load
      lazyLoadSVGs();
    }
  }
  
  // Initialize lazy loading
  initLazyImages();
  initLazySVGs();
  
  // Re-check on window resize and orientation change
  window.addEventListener('resize', function() {
    initLazyImages();
    initLazySVGs();
  });
  window.addEventListener('orientationchange', function() {
    initLazyImages();
    initLazySVGs();
  });
});
