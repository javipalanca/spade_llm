window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Mermaid configuration and theme detection
function configureMermaid() {
  if (typeof mermaid !== 'undefined') {
    // Get current color scheme
    const isDarkMode = document.body.getAttribute('data-md-color-scheme') === 'slate';
    
    // Configure Mermaid with appropriate theme
    mermaid.initialize({
      startOnLoad: true,
      theme: isDarkMode ? 'dark' : 'default',
      themeVariables: isDarkMode ? {
        background: '#1e293b',
        primaryColor: '#334155',
        primaryTextColor: '#e2e8f0',
        primaryBorderColor: '#64748b',
        lineColor: '#64748b',
        secondaryColor: '#475569',
        tertiaryColor: '#334155',
        cScale0: '#334155',
        cScale1: '#475569',
        cScale2: '#64748b'
      } : {
        background: '#ffffff',
        primaryColor: '#f8fafc',
        primaryTextColor: '#1f2937',
        primaryBorderColor: '#9333ea',
        lineColor: '#9333ea',
        secondaryColor: '#f3e8ff',
        tertiaryColor: '#e9d5ff',
        cScale0: '#f8fafc',
        cScale1: '#f3e8ff',
        cScale2: '#e9d5ff'
      },
      flowchart: {
        htmlLabels: true,
        curve: 'basis'
      },
      sequence: {
        htmlLabels: true,
        diagramMarginX: 50,
        diagramMarginY: 10,
        actorMargin: 50,
        width: 150,
        height: 65,
        boxMargin: 10,
        boxTextMargin: 5,
        noteMargin: 10,
        messageMargin: 35
      }
    });
    
    // Re-render existing diagrams if they exist
    const diagrams = document.querySelectorAll('.mermaid');
    diagrams.forEach((diagram, index) => {
      if (diagram.getAttribute('data-processed') === 'true') {
        // Reset and re-render
        diagram.removeAttribute('data-processed');
        diagram.innerHTML = diagram.getAttribute('data-original') || diagram.innerHTML;
        mermaid.init(undefined, diagram);
      }
    });
  }
}

// Initialize on page load
document$.subscribe(() => { 
  MathJax.typesetPromise();
  
  // Configure Mermaid after a short delay to ensure it's loaded
  setTimeout(configureMermaid, 100);
});

// Reconfigure Mermaid when color scheme changes
if (typeof MutationObserver !== 'undefined') {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'attributes' && 
          mutation.attributeName === 'data-md-color-scheme') {
        setTimeout(configureMermaid, 100);
      }
    });
  });
  
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ['data-md-color-scheme']
  });
}
