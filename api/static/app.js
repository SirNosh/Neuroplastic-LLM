document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatButton = document.getElementById('clear-chat');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const maxTokensSelect = document.getElementById('max-tokens');
    const loraIndicator = document.getElementById('lora-indicator');
    
    // API settings
    const API_URL = '/v1/generate';
    
    // Chat history
    let chatHistory = [];
    
    // Settings
    let settings = {
        temperature: 0.7,
        maxTokens: 512,
        loraId: null
    };
    
    // Initialize
    function init() {
        setupEventListeners();
        userInput.focus();
        
        // Check for available LoRA adapters
        fetchLoraAdapters();
    }
    
    // Setup event listeners
    function setupEventListeners() {
        // Send message on button click
        sendButton.addEventListener('click', sendMessage);
        
        // Send message on Enter key (Shift+Enter for new line)
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
            
            // Auto-resize textarea
            setTimeout(() => {
                userInput.style.height = 'auto';
                userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
            }, 0);
        });
        
        // Clear chat history
        clearChatButton.addEventListener('click', () => {
            clearChat();
        });
        
        // Update temperature value
        temperatureSlider.addEventListener('input', () => {
            settings.temperature = parseFloat(temperatureSlider.value);
            temperatureValue.textContent = settings.temperature;
        });
        
        // Update max tokens
        maxTokensSelect.addEventListener('change', () => {
            settings.maxTokens = parseInt(maxTokensSelect.value);
        });
    }
    
    // Send message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to UI
        addMessageToUI('user', message);
        
        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';
        userInput.focus();
        
        // Add typing indicator
        const typingIndicator = addTypingIndicator();
        
        try {
            // Call API to get response
            const response = await generateResponse(message);
            
            // Remove typing indicator
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            // Add assistant response to UI
            addMessageToUI('assistant', response);
            
            // Save to history
            chatHistory.push({ role: 'user', content: message });
            chatHistory.push({ role: 'assistant', content: response });
            
            // Scroll to bottom
            scrollToBottom();
        } catch (error) {
            // Remove typing indicator
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            // Show error message
            addMessageToUI('system', `Error: ${error.message || 'Failed to generate response'}`);
            console.error('Error generating response:', error);
        }
    }
    
    // Generate response from API
    async function generateResponse(prompt) {
        const requestData = {
            prompt: prompt,
            max_tokens: settings.maxTokens,
            temperature: settings.temperature,
            top_p: 0.9
        };
        
        // Add LoRA ID if selected
        if (settings.loraId) {
            requestData.lora_id = settings.loraId;
        }
        
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }
        
        const data = await response.json();
        return data.response;
    }
    
    // Add message to UI
    function addMessageToUI(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Convert markdown-like content (basic support)
        const formattedContent = formatMessage(content);
        messageContent.innerHTML = formattedContent;
        
        messageDiv.appendChild(messageContent);
        
        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'timestamp';
        timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        messageDiv.appendChild(timestamp);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the new message
        scrollToBottom();
        
        return messageDiv;
    }
    
    // Add typing indicator
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'typing-indicator';
        typingContent.innerHTML = 'Thinking<span></span><span></span><span></span>';
        
        typingDiv.appendChild(typingContent);
        chatMessages.appendChild(typingDiv);
        
        scrollToBottom();
        
        return typingDiv;
    }
    
    // Format message (basic markdown support)
    function formatMessage(content) {
        // Replace newlines with <br>
        let formatted = content.replace(/\n/g, '<br>');
        
        // Code blocks (```code```)
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>');
        
        // Inline code (`code`)
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Bold (**text**)
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic (*text*)
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        return formatted;
    }
    
    // Scroll chat to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Clear chat
    function clearChat() {
        // Remove all messages except the first system message
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        // Clear chat history
        chatHistory = [];
    }
    
    // Fetch available LoRA adapters
    async function fetchLoraAdapters() {
        try {
            const response = await fetch('/v1/lora');
            
            if (response.ok) {
                const data = await response.json();
                
                if (data && data.adapters && data.adapters.length > 0) {
                    updateLoraIndicator(data.adapters.length);
                }
            }
        } catch (error) {
            console.error('Error fetching LoRA adapters:', error);
        }
    }
    
    // Update LoRA indicator
    function updateLoraIndicator(count) {
        loraIndicator.textContent = `LoRA: ${count} available`;
        
        // If we want to add a dropdown to select LoRA adapters, we could do that here
    }
    
    // Initialize
    init();
}); 