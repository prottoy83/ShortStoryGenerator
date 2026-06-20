const SERVER_URL = 'http://localhost:8000';
const messagesEl = document.getElementById('messages');
const composer = document.getElementById('composer');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('sendBtn');
const tokenSize = document.getElementById('tokenSize');

function el(tag, cls, text){
  const e = document.createElement(tag);
  if(cls) e.className = cls;
  if(text !== undefined) e.textContent = text;
  return e;
}

function addMessage(role, text){
  const m = el('div', `message ${role}`);
  m.textContent = text;
  messagesEl.appendChild(m);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setTyping(on){
  if(on){
    const t = el('div','message assistant typing');
    t.textContent = 'Assistant is typing...';
    t.id = 'typingIndicator';
    messagesEl.appendChild(t);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else {
    const t = document.getElementById('typingIndicator');
    if(t) t.remove();
  }
}

composer.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const text = promptEl.value.trim();
  if(!text) return;
  addMessage('user', text);
  promptEl.value = '';
  setTyping(true);

  // Call FastAPI server to generate story
  const tokens = tokenSize ? Number(tokenSize.value || 150) : 150;
  try {
    sendBtn.disabled = true;
    const res = await fetch(`${SERVER_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: text, token: tokens })
    });

    if(!res.ok){
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    const story = data.message || data.Message || 'No response from server.';
    setTyping(false);
    addMessage('assistant', story);
  } catch(err){
    setTyping(false);
    addMessage('assistant', `Error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
  }
});

// send on Enter (shift+enter for newline)
promptEl.addEventListener('keydown', (e)=>{
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    sendBtn.click();
  }
});

// sample welcome message
addMessage('assistant', 'Hello — I can help generate short stories. Type a prompt and press Send.');
