// Lazy loading implementation for images, videos, and SVGs
document.addEventListener('DOMContentLoaded', function() {
  // Throttle function to limit how often a function can run
  function throttle(callback, limit) {
    let waiting = false;
    return function() {
      if (!waiting) {
        callback.apply(this, arguments);
        waiting = true;
        setTimeout(function() {
          waiting = false;
        }, limit);
      }
    };
  }

  // Initialize lazy SVGs
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
      }, {
        rootMargin: '400px 0px', // Loads SVGs 400px before they enter the viewport
        threshold: 0.1 // Triggers when 10% of the SVG is visible
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
            document.removeEventListener('scroll', throttledLazyLoadSVGs);
            window.removeEventListener('resize', throttledLazyLoadSVGs);
            window.removeEventListener('orientationChange', throttledLazyLoadSVGs);
          }
        }, 20);
      }
      
      // Apply throttling to scroll event handler for better performance
      const throttledLazyLoadSVGs = throttle(lazyLoadSVGs, 100);
      
      document.addEventListener('scroll', throttledLazyLoadSVGs);
      window.addEventListener('resize', throttledLazyLoadSVGs);
      window.addEventListener('orientationChange', throttledLazyLoadSVGs);
      
      // Initial load
      lazyLoadSVGs();
    }
  }
// Initialize lazy loading for images
  function initLazyImages() {
    const lazyImages = document.querySelectorAll('img.lazy-image[data-src]');
    
    // For Core Web Vitals optimization, immediately load images that are currently visible
    function loadVisibleImagesImmediately() {
      lazyImages.forEach(function(image) {
        if (isInViewport(image)) {
          processImage(image);
        }
      });
    }
    
    // Helper function to check if an element is in the viewport
    function isInViewport(element) {
      const rect = element.getBoundingClientRect();
      return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
      );
    }

    if ('IntersectionObserver' in window) {
      // Function for processing images
      function processImage(image) {
        if (image.dataset.src) {
          const picture = image.closest('picture');

          // Process source tags first
          if (picture) {
            const sources = picture.querySelectorAll('source[data-srcset]');

                // Process all source elements first
                Promise.all(Array.from(sources).map(function(source) {
                  return new Promise(function(resolve) {
                    source.srcset = source.dataset.srcset;
                    source.removeAttribute('data-srcset');
                    // Resolve immediately as modern browsers can handle this efficiently
                    resolve();
                  });
                })).then(function() {
                  // Only after all sources are processed, set the img src
                  image.src = image.dataset.src;
                  image.removeAttribute('data-src');

              // Add loaded class when image is loaded
              image.onload = function() {
                image.classList.add('loaded');
                if (picture) {
                  picture.classList.add('loaded');
                }
              };
            });
          } else {
            // No picture parent, process the image normally
            image.src = image.dataset.src;
            image.removeAttribute('data-src');

            // Add loaded class when image is loaded
            image.onload = function() {
              image.classList.add('loaded');
            };
          }
        }
      }
      
      // Observer for images
      const imageObserver = new IntersectionObserver(function(entries, observer) {
        entries.forEach(function(entry) {
          if (entry.isIntersecting) {
            const image = entry.target;
            processImage(image);
            imageObserver.unobserve(image);
          }
        });
      }, {
        rootMargin: '400px 0px', // Loads images 400px before they enter the viewport
        threshold: 0.1 // Triggers when 10% of the image is visible
      });

      // Register images with the observer
      lazyImages.forEach(function(image) {
        imageObserver.observe(image);
      });
      
      // Immediately load visible images to improve LCP (Largest Contentful Paint)
      loadVisibleImagesImmediately();
    } else {
      // Fallback for browsers that don't support IntersectionObserver
      let lazyImageTimeout;

      function lazyLoadImages() {
        if (lazyImageTimeout) {
          clearTimeout(lazyImageTimeout);
        }

        lazyImageTimeout = setTimeout(function() {
          const scrollTop = window.pageYOffset;
          
          // Loading images
          lazyImages.forEach(function(image) {
            if (image.offsetTop < (window.innerHeight + scrollTop + 200)) { // Standard margin
              if (image.dataset.src) {
                const picture = image.closest('picture');

                // Process source tags first
                if (picture) {
                  const sources = picture.querySelectorAll('source[data-srcset]');

                  // Process all source elements first
                  Promise.all(Array.from(sources).map(function(source) {
                    return new Promise(function(resolve) {
                      source.srcset = source.dataset.srcset;
                      source.removeAttribute('data-srcset');
                      // Resolve immediately as modern browsers can handle this efficiently
                      resolve();
                    });
                  })).then(function() {
                    // Only after all sources are processed, set the img src
                    image.src = image.dataset.src;
                    image.removeAttribute('data-src');

                    // Add loaded class when image is loaded
                    image.onload = function() {
                      image.classList.add('loaded');
                      if (picture) {
                        picture.classList.add('loaded');
                      }
                    };
                  });
                } else {
                  // No picture parent, process the image normally
                  image.src = image.dataset.src;
                  image.removeAttribute('data-src');

                  // Add loaded class when image is loaded
                  image.onload = function() {
                    image.classList.add('loaded');
                  };
                }
              }
            }
          });
          
          // Remove event listeners if all images are loaded
          if (lazyImages.length === 0) {
            document.removeEventListener('scroll', throttledLazyLoadImages);
            window.removeEventListener('resize', throttledLazyLoadImages);
            window.removeEventListener('orientationChange', throttledLazyLoadImages);
          }
        }, 20);
      }

      // Apply throttling to scroll event handler for better performance
      const throttledLazyLoadImages = throttle(lazyLoadImages, 100);

      document.addEventListener('scroll', throttledLazyLoadImages);
      window.addEventListener('resize', throttledLazyLoadImages);
      window.addEventListener('orientationChange', throttledLazyLoadImages);

      // Initial load
      lazyLoadImages();
      
      // Immediately load visible images to improve LCP (Largest Contentful Paint)
      loadVisibleImagesImmediately();
    }
  }
  
  // Add loaded class to all images when they finish loading
  function initImageLoadedClass() {
    const lazyImages = document.querySelectorAll('img.lazy-image:not([data-src])');
    const lazyPictures = document.querySelectorAll('picture.lazy-picture');
    
    // Add loaded class to images when they finish loading
    lazyImages.forEach(function(image) {
      // If the image is already loaded
      if (image.complete) {
        image.classList.add('loaded');
        
        // Find parent picture element and add loaded class
        const picture = image.closest('picture');
        if (picture) {
          picture.classList.add('loaded');
        }
      } else {
        // Add load event listener
        image.addEventListener('load', function() {
          image.classList.add('loaded');
          
          // Find parent picture element and add loaded class
          const picture = image.closest('picture');
          if (picture) {
            picture.classList.add('loaded');
          }
        });
      }
    });
  }
  
  // Initialize lazy loading
  initLazySVGs();
  initLazyImages();
  initImageLoadedClass();
});
