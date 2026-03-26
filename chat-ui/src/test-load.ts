import { loadLiteRt, loadAndCompile } from '@litertjs/core';

async function testLoad() {
  try {
    console.log('Testing LiteRT.js load...');
    await loadLiteRt('https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/');
    const model = await loadAndCompile('/models/model.litertlm', {
      accelerator: 'webgpu',
    });
    console.log('Model loaded successfully!', model);
    console.log('Signatures:', model.getSignatureNames());
  } catch (err) {
    console.error('Model load failed:', err);
  }
}

testLoad();
