// StrawCore AI OpenClaw Integration Script
// This script provides a simple interface for the OpenClaw agent to communicate
// with the locally hosted Manglish LLM (via Ollama or an exposed API endpoint).

const axios = require('axios');

class StrawCoreManglishAgent {
    constructor(modelName = "strawcore-manglish", endpoint = "http://localhost:11434/api/chat") {
        this.modelName = modelName;
        this.endpoint = endpoint;
    }

    /**
     * Sends a unified prompt (English, Manglish, or Malayalam) to the LLM and returns the agentic action or response.
     * @param {string} userInput - The raw client input.
     * @param {string} context - Current conversation or workflow state (e.g., {"booking_step": "waiting_date"})
     * @returns {string} The LLM's response
     */
    async handleClientRequest(userInput, context = {}) {
        const systemPrompt = `You are StrawCore AI. Current workflow context: ${JSON.stringify(context)}. Respond accurately to the client in the appropriate language (Manglish or English).`;
        
        try {
            const response = await axios.post(this.endpoint, {
                model: this.modelName,
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userInput }
                ],
                stream: false
            });

            return response.data.message.content;
        } catch (error) {
            console.error("OpenClaw Error - Failed to connect to Manglish LLM:", error.message);
            return "Kshemikkanam, currently the system is facing a technical issue. (Sorry, there is a technical issue).";
        }
    }
}

module.exports = StrawCoreManglishAgent;

// Usage Example (for OpenClaw main loop):
// const agent = new StrawCoreManglishAgent();
// agent.handleClientRequest("Enikku kochi yil oru room venam", { intent: "hotel_booking" }).then(console.log);
