/**
 * Dynamic Font Sizing for H1 Elements
 * Client-side enhancement for responsive font sizing
 * Works in conjunction with server-side font class mapping
 */

class DynamicFontSizing {
    constructor() {
        // All possible font size classes that could be applied
        this.fontClasses = [
            'lg:text-[1.5rem]',
            'lg:text-[1.6rem]', 'xl:text-[1.75rem]', 
            'lg:text-[1.75rem]', 'xl:text-[2rem]',
            'lg:text-[2rem]', 'xl:text-[2.25rem]',
            'lg:text-[2.5rem]', 'xl:text-[2.75rem]',
            'lg:text-[3rem]', 'xl:text-[3.25rem]',
            'lg:text-[3.5rem]', 'xl:text-[3.75rem]',
            'lg:text-6xl', 'text-5xl'
        ];
        
        this.breakpoints = [
            { max: 120, classes: ['lg:text-[1.5rem]'] },
            { max: 100, classes: ['lg:text-[1.6rem]', 'xl:text-[1.75rem]'] },
            { max: 85, classes: ['lg:text-[1.75rem]', 'xl:text-[2rem]'] },
            { max: 70,  classes: ['lg:text-[2rem]', 'xl:text-[2.25rem]'] },
            { max: 60,  classes: ['lg:text-[2.5rem]', 'xl:text-[2.75rem]'] },
            { max: 50,  classes: ['lg:text-[3rem]', 'xl:text-[3.25rem]'] },
            { max: 40,  classes: ['lg:text-[3.5rem]', 'xl:text-[3.75rem]'] },
            { max: 0,   classes: ['lg:text-6xl'] },
        ];
        
        // Tracking classes that might conflict
        this.trackingClasses = ['tracking-[-2.832px]', 'tracking-tight'];
        this.init();
    }

    init() {
        this.elements = document.querySelectorAll('[data-dynamic-font]');
        if (!this.elements.length) return;

        this.apply();

        window.addEventListener('resize', this.debounce(() => this.apply(), 250));
    }

    apply() {
        this.elements.forEach(el => this.adjustFontSize(el));
    }

    adjustFontSize(el) {
        const textLength = el.textContent.trim().length;
        const rule = this.breakpoints.find(b => textLength > b.max) || this.breakpoints.at(-1);

        this.removeFontClasses(el);
        el.classList.add(...rule.classes);
    }

    removeFontClasses(el) {
        el.classList.remove(...this.fontClasses, ...this.trackingClasses);
    }

    debounce(fn, delay) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => fn(...args), delay);
        };
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => new DynamicFontSizing());
