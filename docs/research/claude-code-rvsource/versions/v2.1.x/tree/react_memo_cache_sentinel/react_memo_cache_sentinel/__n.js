// Module: __n

const texts = [];

const composed_value = texts.join('\\n'); /* confidence: 30% */

function cmdcheckboxchecked_n_copyallbt() {
  
      const checkboxes = document.querySelectorAll('.cmd-checkbox:checked');
  
      const texts = [];
  
      checkboxes.forEach(cb => {
    
        if (cb.dataset.text) {
       texts.push(cb.dataset.text);
       
    }
      
  });
  
      const combined = texts.join('\\n');
  
      const btn = document.querySelector('.copy-all-btn');
  
      if (btn) {
    
        navigator.clipboard.writeText(combined).then(() => {
      
          btn.textContent = 'Copied ' + texts.length + ' items!';
      
          btn.classList.add('copied');
      
          setTimeout(() => {
         btn.textContent = 'Copy All Checked';
         btn.classList.remove('copied');
         
      }, 2000);
      
        
    });
    
      
  }
    
} /* confidence: 65% */

