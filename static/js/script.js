document.addEventListener('DOMContentLoaded', function() {
    // Handle form submission and show loading spinner
    const form = document.querySelector('form[action="/predict"]');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (form && loadingOverlay) {
        form.addEventListener('submit', function() {
            if (form.getAttribute('method') === 'post') {
                // Check if a file was selected before showing the loading spinner
                const fileInput = document.querySelector('input[type="file"]');
                if (fileInput && fileInput.files.length > 0) {
                    loadingOverlay.classList.remove('hidden');
                }
            }
        });
    }
    
    // Typing effect for the home page
    const typingElement = document.querySelector('.typing');
    if (typingElement) {
        const texts = [
            "Detect deepfakes with AI",
            "Verify image authenticity",
            "Powered by MesoNet"
        ];
        
        let textIndex = 0;
        let charIndex = 0;
        let isDeleting = false;
        let typingSpeed = 100;
        
        // Clear any existing content
        typingElement.textContent = '';
        
        function type() {
            const currentText = texts[textIndex];
            
            if (isDeleting) {
                typingElement.textContent = currentText.substring(0, charIndex);
                charIndex--;
            } else {
                typingElement.textContent = currentText.substring(0, charIndex + 1);
                charIndex++;
            }
            
            // Adjust typing speed for more natural effect
            let nextSpeed = typingSpeed;
            if (!isDeleting && charIndex === currentText.length) {
                // Pause at the end of typing
                isDeleting = true;
                nextSpeed = 1500; // Wait longer at the end of the sentence
            } else if (isDeleting && charIndex === 0) {
                // Switch to the next text
                isDeleting = false;
                textIndex = (textIndex + 1) % texts.length;
                nextSpeed = 500; // Pause before starting the next sentence
            } else if (isDeleting) {
                nextSpeed = 50; // Delete faster
            } else {
                // Random variation in typing speed for natural effect
                nextSpeed = Math.random() * 50 + 80;
            }
            
            setTimeout(type, nextSpeed);
        }
        
        // Start typing effect
        setTimeout(type, 1000);
    }
});
