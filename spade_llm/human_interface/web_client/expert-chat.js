/**
 * SPADE LLM Human Expert Interface
 *
 * This web client allows human experts to receive and respond to queries
 * from SPADE LLM agents via XMPP.
 */

// Global variables
let xmppClient = null;
let queries = new Map(); // Store queries by ID
let expertJid = null;

// Get DOM elements
const loginForm = document.getElementById('loginForm');
const mainInterface = document.getElementById('mainInterface');
const connectForm = document.getElementById('connectForm');
const queriesList = document.getElementById('queriesList');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const debugLog = document.getElementById('debugLog');
const showAnsweredCheckbox = document.getElementById('showAnswered');
const clearAnsweredBtn = document.getElementById('clearAnswered');

// Debug logging
function debug(message, type = 'info') {
    console.log(`[${type}] ${message}`);
    if (debugLog) {
        const entry = document.createElement('div');
        entry.className = `debug-entry debug-${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        debugLog.appendChild(entry);
        debugLog.scrollTop = debugLog.scrollHeight;
    }
}

// Update connection status
function updateConnectionStatus(status, text) {
    statusIndicator.className = `status-indicator status-${status}`;
    statusText.textContent = text;
}

// Parse query from message body
function parseQuery(body) {
    // Expected format: [Query ID] Question\n\nContext: context\n\n(Please reply...)
    const queryMatch = body.match(/\[Query ([^\]]+)\] (.+?)(?:\n\nContext: (.+?))?(?:\n\n\(Please reply|$)/s);

    if (queryMatch) {
        return {
            shortId: queryMatch[1],
            question: queryMatch[2].trim(),
            context: queryMatch[3] ? queryMatch[3].trim() : null
        };
    }

    // Fallback: treat entire body as question
    return {
        shortId: 'unknown',
        question: body,
        context: null
    };
}

// Create query card from template
function createQueryCard(query) {
    const template = document.getElementById('queryTemplate');
    const card = template.content.cloneNode(true);

    const cardElement = card.querySelector('.query-card');
    cardElement.dataset.queryId = query.id;

    // Fill in query details
    card.querySelector('.query-id').textContent = `Query ${query.shortId}`;
    card.querySelector('.query-from').textContent = query.from;
    card.querySelector('.query-time').textContent = query.timestamp.toLocaleTimeString();
    card.querySelector('.query-status').textContent = 'Pending';
    card.querySelector('.query-question').textContent = query.question;

    if (query.context) {
        card.querySelector('.query-context').innerHTML = `<strong>Context:</strong> ${query.context}`;
    } else {
        card.querySelector('.query-context').style.display = 'none';
    }

    // Add event listener for send button
    const sendBtn = card.querySelector('.send-response');
    const responseInput = card.querySelector('.response-input');

    sendBtn.addEventListener('click', () => {
        const response = responseInput.value.trim();
        if (response) {
            sendResponse(query, response);
        }
    });

    // Allow Enter key to send (Shift+Enter for new line)
    responseInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const response = responseInput.value.trim();
            if (response) {
                sendResponse(query, response);
            }
        }
    });

    return cardElement;
}

// Add new query to the UI
function addQuery(stanza) {
    let queryId = stanza.attrs.thread;
    const body = stanza.getChildText('body');

    // If no thread ID in attributes, try to extract from metadata
    if (!queryId) {
        const metadata = stanza.getChild('x', 'jabber:x:data');
        if (metadata) {
            const threadField = metadata.getChildren('field').find(f => f.attrs.var === 'query_id');
            if (threadField) {
                queryId = threadField.getChildText('value');
            }
        }
    }

    // If still no queryId, extract from body
    if (!queryId && body) {
        const match = body.match(/\[Query ([^\]]+)\]/);
        if (match) {
            queryId = match[1];
        }
    }

    if (!queryId || !body) {
        debug(`Received message without thread ID or body. Thread: ${queryId}, Body: ${body ? 'present' : 'missing'}`, 'warn');
        return;
    }

    // Skip if we already have this query
    if (queries.has(queryId)) {
        debug(`Duplicate query received: ${queryId}`, 'warn');
        return;
    }

    const parsed = parseQuery(body);
    const query = {
        id: queryId,
        shortId: parsed.shortId,
        question: parsed.question,
        context: parsed.context,
        from: stanza.attrs.from,
        timestamp: new Date(),
        answered: false,
        response: null
    };

    queries.set(queryId, query);

    // Remove "no queries" message if present
    const noQueries = queriesList.querySelector('.no-queries');
    if (noQueries) {
        noQueries.remove();
    }

    // Add query card to UI
    const card = createQueryCard(query);
    queriesList.insertBefore(card, queriesList.firstChild);

    debug(`New query received: ${query.shortId} from ${query.from}`);

    // Optional: Show notification
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('New Query', {
            body: `${query.shortId}: ${query.question.substring(0, 50)}...`,
            icon: '/favicon.ico'
        });
    }
}

// Send response to query
async function sendResponse(query, responseText) {
    if (!xmppClient) {
        debug('Cannot send response: not connected', 'error');
        return;
    }

    try {
        const { xml } = window.XMPP;
        const message = xml(
            'message',
            {
                to: query.from,
                type: 'chat',
                thread: query.id
            },
            xml('body', {}, responseText)
        );

        await xmppClient.send(message);

        // Update query state
        query.answered = true;
        query.response = responseText;
        queries.set(query.id, query);

        // Update UI
        const card = document.querySelector(`[data-query-id="${query.id}"]`);
        if (card) {
            card.classList.add('answered');
            card.querySelector('.query-status').textContent = 'Answered';
            card.querySelector('.query-response').style.display = 'none';
            card.querySelector('.query-answered').style.display = 'block';
            card.querySelector('.answered-response').textContent = responseText;
        }

        debug(`Response sent for query ${query.shortId}`);

    } catch (error) {
        debug(`Failed to send response: ${error.message}`, 'error');
    }
}

// Connect to XMPP server
async function connect(credentials) {
    debug('Starting connection function');

    if (!window.XMPP) {
        debug('XMPP library not loaded!', 'error');
        updateConnectionStatus('error', 'XMPP library not loaded');
        return;
    }

    const { client } = window.XMPP;
    debug('XMPP client function found');

    try {
        updateConnectionStatus('connecting', 'Connecting...');
        debug('Creating XMPP client...');

        const clientConfig = {
            service: credentials.service,
            domain: credentials.jid.split('@')[1],
            username: credentials.jid.split('@')[0],
            password: credentials.password
        };
        debug(`Client config: ${JSON.stringify({...clientConfig, password: '***'})}`);

        xmppClient = client(clientConfig);
        expertJid = credentials.jid;
        debug('XMPP client created successfully');

        // Set up event handlers
        xmppClient.on('error', (err) => {
            debug(`Error: ${err.message}`, 'error');
            updateConnectionStatus('error', 'Error');
        });

        xmppClient.on('offline', () => {
            debug('Disconnected from server');
            updateConnectionStatus('offline', 'Disconnected');
            loginForm.style.display = 'block';
            mainInterface.style.display = 'none';
        });

        xmppClient.on('online', (address) => {
            debug(`Connected as ${address.toString()}`);
            updateConnectionStatus('online', `Connected as ${address.toString()}`);
            loginForm.style.display = 'none';
            mainInterface.style.display = 'block';

            // Send presence
            xmppClient.send(window.XMPP.xml('presence'));
        });

        xmppClient.on('stanza', (stanza) => {
            debug(`Received stanza: ${stanza.toString()}`, 'debug');

            if (stanza.is('message') && stanza.attrs.type === 'chat') {
                addQuery(stanza);
            }
        });

        // Start connection
        await xmppClient.start();

    } catch (error) {
        debug(`Connection failed: ${error.message}`, 'error');
        updateConnectionStatus('error', 'Connection failed');
    }
}

// Event Listeners
connectForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    debug('Connect form submitted');

    const credentials = {
        service: document.getElementById('service').value,
        jid: document.getElementById('jid').value,
        password: document.getElementById('password').value
    };

    debug(`Attempting connection with: ${JSON.stringify({...credentials, password: '***'})}`);

    await connect(credentials);
});

showAnsweredCheckbox.addEventListener('change', (e) => {
    const answered = document.querySelectorAll('.query-card.answered');
    answered.forEach(card => {
        card.style.display = e.target.checked ? 'block' : 'none';
    });
});

clearAnsweredBtn.addEventListener('click', () => {
    const answered = document.querySelectorAll('.query-card.answered');
    answered.forEach(card => {
        const queryId = card.dataset.queryId;
        queries.delete(queryId);
        card.remove();
    });

    if (queries.size === 0) {
        queriesList.innerHTML = '<p class="no-queries">No queries yet. Waiting for agents to ask questions...</p>';
    }
});

// Request notification permission
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}

// Show debug section in development
if (window.location.hostname === 'localhost') {
    document.getElementById('debugSection').style.display = 'block';
}

// Initialize
debug('Human Expert Interface loaded');
