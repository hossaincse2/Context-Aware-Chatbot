// Additional JavaScript functionality
class ChatbotUI {
    constructor() {
        this.messageCount = 0;
        this.userId = 'user_' + Math.random().toString(36).substr(2, 9);
        this.isTyping = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadStats();
        this.setupAutoRefresh();
    }

    setupEventListeners() {
        // File upload handling
        document.getElementById('file-upload')?.addEventListener('change', this.handleFileUpload.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.sendMessage();
            }
            if (e.ctrlKey && e.key === 'l') {
                e.preventDefault();
                this.clearHistory();
            }
        });
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (file.type === 'text/plain') {
            const content = await file.text();
            document.getElementById('document-content').value = content;
        } else {
            alert('Only text files are supported currently.');
        }
    }

    setupAutoRefresh() {
        setInterval(() => {
            this.loadStats();
        }, 30000);
    }

    formatMessage(message) {
        // Enhanced message formatting
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded">$1</code>')
            .replace(/\n/g, '<br>');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
            type === 'success' ? 'bg-green-500 text-white' :
            type === 'error' ? 'bg-red-500 text-white' :
            'bg-blue-500 text-white'
        }`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    exportChat() {
        const messages = document.querySelectorAll('#chat-messages .message-animation');
        let chatData = [];
        
        messages.forEach(msg => {
            const isUser = msg.classList.contains('justify-end');
            const content = msg.querySelector('div').textContent;
            const timestamp = msg.querySelector('.text-xs').textContent;
            
            chatData.push({
                sender: isUser ? 'user' : 'bot',
                message: content,
                timestamp: timestamp
            });
        });
        
        const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_history.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}
