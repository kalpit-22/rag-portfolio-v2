// Generate a unique session ID for this user session
const sessionId = crypto.randomUUID();
let chatHistory = [];

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const welcomeScreen = document.getElementById('welcome-screen');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const suggestionBtns = document.querySelectorAll('.suggestion-btn');

// --- CHAT LOGIC ---

function createMessageElement(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role);

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const msgContent = document.createElement('div');
    msgContent.classList.add('message-content');
    
    // Basic Markdown formatting (bold, links, code blocks) - simplistic version
    let formattedContent = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
    
    msgContent.innerHTML = formattedContent;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(msgContent);

    return msgDiv;
}

function showTypingIndicator() {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.classList.add('message', 'bot');
    indicatorDiv.id = 'typing-indicator-container';
    
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = '🤖';

    const typingDiv = document.createElement('div');
    typingDiv.classList.add('typing-indicator');
    for(let i=0; i<3; i++) {
        const dot = document.createElement('div');
        dot.classList.add('dot');
        typingDiv.appendChild(dot);
    }

    indicatorDiv.appendChild(avatar);
    indicatorDiv.appendChild(typingDiv);
    
    chatContainer.appendChild(indicatorDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator-container');
    if (indicator) indicator.remove();
}

function scrollToBottom() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}

async function handleSendMessage(text) {
    if (!text.trim()) return;

    // Hide welcome screen
    if (welcomeScreen) welcomeScreen.style.display = 'none';

    // Add user message to UI and history
    chatContainer.appendChild(createMessageElement('user', text));
    chatHistory.push({ role: 'user', content: text });
    chatInput.value = '';
    scrollToBottom();

    // Disable input
    chatInput.disabled = true;
    sendBtn.disabled = true;
    showTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: text,
                chat_history: chatHistory.slice(0, -1), // Exclude current user query from history
                session_id: sessionId
            })
        });

        if (!response.ok) throw new Error('API request failed');

        const data = await response.json();
        
        removeTypingIndicator();
        chatContainer.appendChild(createMessageElement('bot', data.response));
        chatHistory.push({ role: 'assistant', content: data.response });
        
    } catch (error) {
        removeTypingIndicator();
        const errorMsg = "Agent Offline: Error connecting to server. Check API keys and Network connection.";
        chatContainer.appendChild(createMessageElement('bot', errorMsg));
        console.error(error);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
        scrollToBottom();
    }
}

// Event Listeners for Chat
sendBtn.addEventListener('click', () => handleSendMessage(chatInput.value));
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSendMessage(chatInput.value);
});

suggestionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const text = btn.textContent.split(' ').slice(1).join(' ').replace(/"/g, ''); // Extract text without emoji and quotes
        handleSendMessage(text);
    });
});

// --- FILE UPLOAD LOGIC ---

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        updateDropZoneText();
    }
});

fileInput.addEventListener('change', updateDropZoneText);

function updateDropZoneText() {
    const p = dropZone.querySelector('p');
    if (fileInput.files.length > 0) {
        p.textContent = `${fileInput.files.length} file(s) selected`;
    } else {
        p.textContent = 'Drop files here or click to upload';
    }
}

uploadBtn.addEventListener('click', async () => {
    if (fileInput.files.length === 0) {
        showStatus('Please select a file first.', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('session_id', sessionId);
    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('files', fileInput.files[i]);
    }

    uploadBtn.disabled = true;
    showStatus('Encrypting and indexing in RAM...', 'loading');

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            showStatus('Document indexed successfully! ✅', 'success');
        } else {
            showStatus(`Error: ${result.detail || 'Failed to upload'}`, 'error');
        }
    } catch (error) {
        showStatus('Error uploading files.', 'error');
        console.error(error);
    } finally {
        uploadBtn.disabled = false;
        // Hide success message after 3 seconds
        if(uploadStatus.classList.contains('success')) {
            setTimeout(() => {
                uploadStatus.classList.add('hidden');
            }, 3000);
        }
    }
});

function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message ${type}`;
    uploadStatus.classList.remove('hidden');
}
