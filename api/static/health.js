document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const refreshButton = document.getElementById('refresh-data');
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const modelName = document.getElementById('model-name');
    const engineType = document.getElementById('engine-type');
    const modelLoaded = document.getElementById('model-loaded');
    const lastCheck = document.getElementById('last-check');
    const loraCount = document.getElementById('lora-count');
    const loraList = document.getElementById('lora-list');
    const totalRequests = document.getElementById('total-requests');
    const avgLatency = document.getElementById('avg-latency');
    const tokensPerSecond = document.getElementById('tokens-per-second');
    const memoryUsage = document.getElementById('memory-usage');
    const runTestButton = document.getElementById('run-test');
    const testResult = document.getElementById('test-result');

    // Initialize
    function init() {
        setupEventListeners();
        fetchHealthData();
        fetchLoraAdapters();
    }

    // Setup event listeners
    function setupEventListeners() {
        refreshButton.addEventListener('click', () => {
            refreshButton.classList.add('fa-spin');
            Promise.all([fetchHealthData(), fetchLoraAdapters()])
                .finally(() => {
                    setTimeout(() => {
                        refreshButton.classList.remove('fa-spin');
                    }, 500);
                });
        });

        runTestButton.addEventListener('click', runTestGeneration);
    }

    // Fetch health data
    async function fetchHealthData() {
        try {
            const response = await fetch('/v1/health');
            
            if (!response.ok) {
                throw new Error('Health check failed');
            }
            
            const data = await response.json();
            updateHealthDisplay(data);
            
            return data;
        } catch (error) {
            console.error('Error fetching health data:', error);
            showError();
        }
    }

    // Fetch LoRA adapters
    async function fetchLoraAdapters() {
        try {
            const response = await fetch('/v1/lora');
            
            if (!response.ok) {
                throw new Error('Failed to fetch LoRA adapters');
            }
            
            const data = await response.json();
            updateLoraDisplay(data);
            
            return data;
        } catch (error) {
            console.error('Error fetching LoRA adapters:', error);
            loraList.innerHTML = '<div class="loading-placeholder">Failed to load adapters</div>';
        }
    }

    // Run test generation
    async function runTestGeneration() {
        try {
            runTestButton.disabled = true;
            testResult.innerHTML = '<div class="loading-placeholder"><i class="fas fa-circle-notch fa-spin"></i> Running test generation...</div>';
            
            // Send a test prompt to generate endpoint
            const response = await fetch('/v1/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: 'Test prompt: Hello, how are you?',
                    max_tokens: 50,
                    temperature: 0.7,
                }),
            });
            
            if (!response.ok) {
                throw new Error('Test generation failed');
            }
            
            const data = await response.json();
            displayTestResults(data);
        } catch (error) {
            console.error('Error running test generation:', error);
            testResult.innerHTML = `<div class="loading-placeholder">Test failed: ${error.message}</div>`;
        } finally {
            runTestButton.disabled = false;
        }
    }

    // Update health display
    function updateHealthDisplay(data) {
        // Update status indicator
        statusIcon.className = 'status-icon';
        statusIcon.innerHTML = '';
        
        if (data.status === 'healthy') {
            statusIcon.classList.add('healthy');
            statusIcon.innerHTML = '<i class="fas fa-check"></i>';
            statusText.textContent = 'Healthy';
        } else if (data.status === 'degraded') {
            statusIcon.classList.add('degraded');
            statusIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
            statusText.textContent = 'Degraded';
        } else {
            statusIcon.classList.add('unhealthy');
            statusIcon.innerHTML = '<i class="fas fa-times"></i>';
            statusText.textContent = 'Unhealthy';
        }
        
        // Update model info from data or from model_info endpoint if available
        if (data.model_info) {
            modelName.textContent = data.model_info.model_name || 'Unknown';
            engineType.textContent = data.model_info.engine_type || 'Unknown';
        } else {
            // Try to extract from test_generation
            const testGen = data.test_generation || {};
            modelName.textContent = testGen.model || 'Unknown';
            engineType.textContent = data.engine_type || 'vLLM';
        }
        
        // Update model loaded status
        modelLoaded.textContent = data.model_loaded ? 'Yes' : 'No';
        
        // Update last check time
        lastCheck.textContent = new Date().toLocaleString();
        
        // Update performance metrics
        const stats = data.stats || {};
        totalRequests.textContent = stats.total_requests || 0;
        avgLatency.textContent = stats.avg_latency ? stats.avg_latency.toFixed(3) : '0.000';
        
        // Calculate tokens per second if available
        if (stats.total_tokens && stats.total_requests && stats.avg_latency) {
            const tps = stats.total_tokens / (stats.total_requests * stats.avg_latency);
            tokensPerSecond.textContent = isFinite(tps) ? Math.round(tps) : 0;
        } else {
            tokensPerSecond.textContent = 'N/A';
        }
        
        // GPU memory usage
        if (data.gpu_memory_utilization) {
            memoryUsage.textContent = `${Math.round(data.gpu_memory_utilization * 100)}%`;
        } else {
            memoryUsage.textContent = 'N/A';
        }
    }

    // Update LoRA display
    function updateLoraDisplay(data) {
        const adapters = data.adapters || [];
        loraCount.textContent = adapters.length;
        
        if (adapters.length === 0) {
            loraList.innerHTML = '<div class="loading-placeholder">No LoRA adapters available</div>';
            return;
        }
        
        let loraHtml = '';
        adapters.forEach(adapter => {
            loraHtml += `
                <div class="lora-item">
                    <div class="name">${adapter}</div>
                    <div class="badge">Active</div>
                </div>
            `;
        });
        
        loraList.innerHTML = loraHtml;
    }

    // Display test generation results
    function displayTestResults(data) {
        const promptText = data.prompt || 'Test prompt';
        const responseText = data.response || 'No response generated';
        const stats = data.stats || {};
        
        const html = `
            <div class="test-content">
                <div class="test-prompt">
                    <strong>Prompt:</strong> ${promptText}
                </div>
                <div class="test-response">
                    ${responseText}
                </div>
                <div class="test-stats">
                    <div class="test-stat">
                        <div class="value">${stats.latency ? stats.latency.toFixed(3) : 'N/A'}</div>
                        <div class="label">Latency (s)</div>
                    </div>
                    <div class="test-stat">
                        <div class="value">${stats.tokens_generated || 'N/A'}</div>
                        <div class="label">Tokens</div>
                    </div>
                    <div class="test-stat">
                        <div class="value">${stats.tokens_per_second || 'N/A'}</div>
                        <div class="label">Tokens/Sec</div>
                    </div>
                </div>
            </div>
        `;
        
        testResult.innerHTML = html;
    }

    // Show error state
    function showError() {
        statusIcon.className = 'status-icon unhealthy';
        statusIcon.innerHTML = '<i class="fas fa-times"></i>';
        statusText.textContent = 'Connection Error';
        
        modelName.textContent = 'Unknown';
        engineType.textContent = 'Unknown';
        modelLoaded.textContent = 'No';
        lastCheck.textContent = new Date().toLocaleString();
    }

    // Initialize the page
    init();
}); 