import { init as coreInit } from '@cornerstonejs/core';
import { init as dicomImageLoaderInit, wadouri } from '@cornerstonejs/dicom-image-loader';
import * as cornerstoneTools from '@cornerstonejs/tools';

let initialized = false;
let initPromise: Promise<void> | null = null;

export default async function initCornerstone() {
  if (initialized) return;
  if (initPromise) return initPromise;
  
  console.log('Starting Cornerstone3D initialization...');
  
  initPromise = (async () => {
    // Initialize Cornerstone3D Core
    await coreInit();
    console.log('Core initialized');
    
    // Initialize DICOM Image Loader
    await dicomImageLoaderInit({
      maxWebWorkers: navigator.hardwareConcurrency || 1,
    });
    
    // Explicitly register wadouri if needed (some versions require this for volume metadata)
    if (wadouri && typeof wadouri.register === 'function') {
        wadouri.register();
    }
    
    console.log('DICOM Image Loader initialized');
    
    // Initialize Tools
    cornerstoneTools.init();
    console.log('Tools initialized');
    
    initialized = true;
    console.log('Cornerstone3D Initialization Complete');
  })();

  try {
    await initPromise;
  } catch (error) {
    initPromise = null;
    console.error('Failed to initialize Cornerstone3D:', error);
    throw error;
  }
}
