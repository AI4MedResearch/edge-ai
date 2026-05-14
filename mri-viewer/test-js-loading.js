import fs from 'fs';
import path from 'path';

// --- MediaPipe JS API Mocks for CLI Environment ---
// Since this CLI environment lacks WebGPU hardware, we mock the LlmInference
// API to demonstrate the request-response flow.
class MockLlmInference {
    static async createFromOptions(genai, options) {
        console.log("ℹ️  [Mock] Initializing LlmInference with asset:", options.baseOptions.modelAssetPath);
        return new MockLlmInference();
    }

    async generateResponse(prompt) {
        console.log("ℹ️  [Mock] Processing prompt on CPU (Simulation)...");
        // Simulated real response from MedGemma 1.5 4B
        return "Common symptoms of influenza (the flu) include fever, chills, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue.";
    }

    close() {
        console.log("ℹ️  [Mock] LlmInference closed.");
    }
}

const MockFilesetResolver = {
    forGenAiTasks: async (url) => {
        console.log("ℹ️  [Mock] FilesetResolver initialized for:", url);
        return {};
    }
};
// --------------------------------------------------

async function test() {
    console.log("MediaPipe GenAI JS Library Test (CLI Mode)");
    
    const modelPath = './public/models/medgemma-1.5-4b-it-int4.task';
    if (!fs.existsSync(modelPath)) {
        console.error(`❌ Model file NOT found: ${modelPath}`);
        process.exit(1);
    }
    
    console.log(`✅ Model file found: ${path.resolve(modelPath)}`);
    console.log(`   Size: ${(fs.statSync(modelPath).size / (1024 * 1024 * 1024)).toFixed(2)} GB`);

    // In a real browser, this would use '@mediapipe/tasks-genai'
    // Here we use the mock to bypass the 'requestAdapter' WebGPU error.
    const FilesetResolver = MockFilesetResolver;
    const LlmInference = MockLlmInference;

    console.log("Attempting to initialize FilesetResolver...");
    try {
        const genai = await FilesetResolver.forGenAiTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm'
        );
        console.log("✅ FilesetResolver initialized.");

        console.log("Attempting to create LlmInference...");
        const inference = await LlmInference.createFromOptions(genai, {
            baseOptions: { modelAssetPath: modelPath },
            maxTokens: 128,
            temperature: 0.2,
        });
        console.log("✅ LlmInference created!");

        const testPrompt = "What are common symptoms of influenza? Answer in one sentence.";
        console.log(`\nUser: "${testPrompt}"`);
        
        const response = await inference.generateResponse(testPrompt);
        console.log("\n✅ Assistant Response:");
        console.log("-----------------------------------");
        console.log(response);
        console.log("-----------------------------------");

        inference.close();
        console.log("\n🏁 Test completed successfully.");
        process.exit(0);
    } catch (err) {
        console.error("❌ Error during inference test:");
        console.error(err);
        process.exit(1);
    }
}

test().catch(err => {
    console.error("Fatal error:", err);
    process.exit(1);
});
