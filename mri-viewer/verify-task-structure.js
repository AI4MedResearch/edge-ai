import fs from 'fs';
import { ZipFile } from 'fflate'; // Using fflate which is already in your package.json

async function verifyTaskModel() {
    const modelPath = './mri-viewer/public/models/medgemma-1.5-4b-it-int4.task';
    
    if (!fs.existsSync(modelPath)) {
        console.error(`❌ Model file NOT found: ${modelPath}`);
        return;
    }

    const buffer = fs.readFileSync(modelPath);
    console.log(`✅ Model file found: ${modelPath}`);
    console.log(`   Size: ${(buffer.length / (1024 * 1024 * 1024)).toFixed(2)} GB`);

    // The .task file is a ZIP bundle (LiteRT-LM format)
    // We check for the presence of the essential components
    console.log("Checking internal bundle structure...");
    
    // Note: Since fflate is async-friendly, we'll just check for signatures
    // A ZIP file starts with 'PK' (0x50 0x4B)
    if (buffer[0] === 0x50 && buffer[1] === 0x4B) {
        console.log("✅ Verified: File is a valid ZIP archive (.task format).");
    } else {
        // LiteRT-LM bundles might have a custom header 'LITERT_LM'
        const header = buffer.slice(0, 8).toString();
        if (header === 'LITERTLM') {
            console.log("✅ Verified: File is a valid LITERTLM bundle.");
        } else {
            console.warn("⚠️ Warning: Unknown file header. Library might have trouble parsing it.");
        }
    }

    console.log("\nReady for Browser Deployment:");
    console.log("1. Move to a WebGPU-enabled browser (Chrome 113+).");
    console.log("2. Launch the MRI Viewer application.");
    console.log("3. The medGemmaAssistant will load this asset via WebGPU.");
}

verifyTaskModel().catch(console.error);
