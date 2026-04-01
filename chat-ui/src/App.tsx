import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, User, Bot, Loader2, AlertCircle, RefreshCcw, Image as ImageIcon, X } from 'lucide-react';
import { LlmInference, FilesetResolver } from '@mediapipe/tasks-genai';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  imageUrl?: string;
}

// Model Configurations
const MODEL_CONFIGS = {
  MEDGEMMA_1_5: {
    id: 'medgemma-1.5-4b',
    displayName: 'MedGemma 1.5 Edge',
    path: '/models/litertlm_medgemma-1.5-4b-it-fp8.litertlm',
    maxTokens: 2048,
    maxNumImages: 1,
    temperature: 0.7,
    topK: 40,
  },
  // Backup: Gemma 3 configuration
  GEMMA_3: {
    id: 'gemma-3-2b',
    displayName: 'Gemma 3 Multimodal',
    path: '/models/gemma-3n-E2B-it-int4-Web.litertlm',
    maxTokens: 2048,
    maxNumImages: 1,
    temperature: 0.7,
    topK: 40,
  }
};

// Current active model selection
const ACTIVE_CONFIG = MODEL_CONFIGS.MEDGEMMA_1_5;
// To use backup model manually, switch to: 
// const ACTIVE_CONFIG = MODEL_CONFIGS.GEMMA_3;

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const llmInferenceRef = useRef<LlmInference | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const initModel = useCallback(async () => {
    setIsModelLoading(true);
    setModelError(null);
    
    try {
      if (!navigator.gpu) {
        throw new Error('WebGPU is not supported in this browser. Please use Chrome/Edge with WebGPU enabled.');
      }
      
      const genai = await FilesetResolver.forGenAiTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm'
      );
      
      if (llmInferenceRef.current) {
        try {
          llmInferenceRef.current.close();
        } catch (e) {
          console.warn('Error closing previous model instance:', e);
        }
      }

      // Try Primary: MedGemma 1.5 4B
      console.log(`Attempting to load primary: ${MODEL_CONFIGS.MEDGEMMA_1_5.displayName}...`);
      try {
        const model = await LlmInference.createFromOptions(genai, {
          baseOptions: {
            modelAssetPath: `${MODEL_CONFIGS.MEDGEMMA_1_5.path}?v=${Date.now()}`,
          },
          maxTokens: MODEL_CONFIGS.MEDGEMMA_1_5.maxTokens,
          maxNumImages: MODEL_CONFIGS.MEDGEMMA_1_5.maxNumImages,
          topK: MODEL_CONFIGS.MEDGEMMA_1_5.topK,
          temperature: MODEL_CONFIGS.MEDGEMMA_1_5.temperature,
          randomSeed: Math.floor(Math.random() * 1000),
        });
        
        llmInferenceRef.current = model;
        setIsModelLoading(false);
        console.log('MedGemma 1.5 loaded successfully!');
        return; // Success
      } catch (primaryErr) {
        console.warn('MedGemma 1.5 load failed, falling back to Gemma 3:', primaryErr);
        
        // Fallback: Gemma 3 2B
        console.log(`Attempting to load backup: ${MODEL_CONFIGS.GEMMA_3.displayName}...`);
        const model = await LlmInference.createFromOptions(genai, {
          baseOptions: {
            modelAssetPath: `${MODEL_CONFIGS.GEMMA_3.path}?v=${Date.now()}`,
          },
          maxTokens: MODEL_CONFIGS.GEMMA_3.maxTokens,
          maxNumImages: MODEL_CONFIGS.GEMMA_3.maxNumImages,
          topK: MODEL_CONFIGS.GEMMA_3.topK,
          temperature: MODEL_CONFIGS.GEMMA_3.temperature,
          randomSeed: Math.floor(Math.random() * 1000),
        });
        
        llmInferenceRef.current = model;
        setIsModelLoading(false);
        console.log('Gemma 3 (Backup) loaded successfully!');
      }
    } catch (err) {
      console.error('Initialization failed:', err);
      setModelError(err instanceof Error ? err.message : String(err));
      setIsModelLoading(false);
    }
  }, []);

  useEffect(() => {
    initModel();
    return () => {
      llmInferenceRef.current?.close();
    };
  }, [initModel]);

  const handleReset = () => {
    setMessages([]);
    setInput('');
    setSelectedImage(null);
    setPreviewUrl(null);
    initModel();
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const loadImageElement = (file: File): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const buildTurnPrompt = (msg: Message, isFirstTurn: boolean) => {
    let prompt = '';
    if (isFirstTurn) {
      prompt += '<bos>';
    }
    const roleTag = msg.role === 'assistant' ? 'model' : msg.role;
    prompt += `<start_of_turn>${roleTag}\n${msg.content}<end_of_turn>\n`;
    
    if (msg.role === 'user') {
      prompt += '<start_of_turn>model\n';
    }
    return prompt;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!input.trim() && !selectedImage) || isLoading || !llmInferenceRef.current) return;

    const userMessage: Message = { 
      role: 'user', 
      content: input.trim() || (selectedImage ? "Describe this image." : ""),
      imageUrl: previewUrl || undefined
    };
    
    const isFirstTurn = messages.length === 0;
    const currentImage = selectedImage;

    // Add user message and a placeholder for the assistant response
    setMessages((prev) => [...prev, userMessage, { role: 'assistant', content: '' }]);
    
    setInput('');
    setSelectedImage(null);
    setPreviewUrl(null);
    setIsLoading(true);

    try {
      const turnPrompt = buildTurnPrompt(userMessage, isFirstTurn);
      let accumulated = '';
      
      const streamingListener = (partial: string, _done: boolean) => {
        // Smart accumulator: handle both delta and cumulative inputs
        if (accumulated && partial.startsWith(accumulated)) {
          accumulated = partial; // Cumulative
        } else {
          accumulated += partial; // Delta
        }

        setMessages((prev) => {
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          if (updated[lastIndex]?.role === 'assistant') {
            updated[lastIndex] = { ...updated[lastIndex], content: accumulated };
          }
          return updated;
        });
      };

      let finalResult: string;
      if (currentImage) {
        const imgElement = await loadImageElement(currentImage);
        finalResult = await llmInferenceRef.current.generateResponse(
          [{ imageSource: imgElement }, turnPrompt],
          streamingListener
        );
        URL.revokeObjectURL(imgElement.src);
      } else {
        finalResult = await llmInferenceRef.current.generateResponse(turnPrompt, streamingListener);
      }

      // Final synchronization with the authoritative result from the promise
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (updated[lastIndex]?.role === 'assistant') {
          updated[lastIndex] = { ...updated[lastIndex], content: finalResult.trim() };
        }
        return updated;
      });

    } catch (error) {
      console.error("Inference error:", error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (updated[lastIndex]?.role === 'assistant') {
          updated[lastIndex] = { ...updated[lastIndex], content: `Error communicating with model: ${errorMessage}` };
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-800">
      <header className="bg-white shadow-sm py-4 px-6 border-b flex items-center justify-between">
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <Bot className="text-primary w-6 h-6" />
          {ACTIVE_CONFIG.displayName}
        </h1>
        <div className="flex items-center gap-4">
          {isModelLoading && (
            <div className="text-sm text-gray-500 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading Model...
            </div>
          )}
          {!isModelLoading && !modelError && (
            <button 
              onClick={handleReset}
              aria-label="Reset Chat"
              title="Reset Chat"
              className="p-2 text-gray-500 hover:text-primary hover:bg-gray-100 rounded-full transition-colors"
            >
              <RefreshCcw className="w-5 h-5" />
            </button>
          )}
        </div>
      </header>
      
      <main className="flex-1 overflow-y-auto p-4 sm:p-6 w-full max-w-4xl mx-auto flex flex-col gap-4">
        {modelError && (
          <div className="bg-red-50 border border-red-200 text-red-700 p-5 rounded-2xl flex items-start gap-4 shadow-sm">
            <AlertCircle className="w-6 h-6 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-bold text-lg">Model Loading Failed</p>
              <p className="mt-1">{modelError}</p>
              <div className="mt-4 text-sm opacity-80 space-y-2">
                <p>• Ensure you have at least 16GB of RAM available (MedGemma 4B is larger).</p>
                <p>• Verify that WebGPU is enabled in browser flags if using an older version.</p>
                <p>• Make sure the MedGemma model bundle is present in <code>public/models/litertlm_medgemma-1.5-4b-it-fp8.litertlm</code>.</p>
              </div>
              <button 
                onClick={initModel}
                className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
              >
                Retry Loading
              </button>
            </div>
          </div>
        )}

        {messages.length === 0 && !modelError && (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-400">
            <Bot className="w-20 h-20 mb-6 text-gray-200" />
            <div className="text-center">
              <p className="text-xl font-medium text-gray-500">
                {isModelLoading ? 'Preparing Local Engine' : `${ACTIVE_CONFIG.displayName} is Ready`}
              </p>
              <p className="mt-2 max-w-xs mx-auto text-sm leading-relaxed">
                {isModelLoading 
                  ? `We are initializing ${ACTIVE_CONFIG.displayName} directly in your browser.` 
                  : 'Upload a medical image or ask a health question. Everything stays on your device.'}
              </p>
            </div>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div 
            key={idx} 
            className={`flex items-start gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
          >
            <div className={`p-2.5 rounded-full flex-shrink-0 shadow-sm ${msg.role === 'user' ? 'bg-primary text-white' : 'bg-white border border-gray-200 text-gray-600'}`}>
              {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
            </div>
            
            <div className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${msg.role === 'user' ? 'bg-primary text-white rounded-tr-none' : 'bg-white border border-gray-100 text-gray-800 rounded-tl-none'}`}>
              {msg.imageUrl && (
                <img 
                  src={msg.imageUrl} 
                  alt="Uploaded content" 
                  className="max-w-full rounded-lg mb-3 shadow-sm border border-black/5"
                />
              )}
              <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'prose-invert' : 'text-gray-800'}`}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content + (isLoading && idx === messages.length - 1 ? ' ▮' : '')}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex items-start gap-4 flex-row">
            <div className="p-2.5 rounded-full flex-shrink-0 bg-white border border-gray-200 text-gray-600 shadow-sm">
              <Bot className="w-5 h-5" />
            </div>
            <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-none p-4 shadow-sm flex items-center justify-center min-w-[100px]">
              <div className="flex gap-1.5">
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce"></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </main>

      <footer className="p-4 bg-white border-t border-gray-200">
        <div className="max-w-4xl mx-auto mb-4">
          {previewUrl && (
            <div className="relative inline-block group">
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="h-24 w-24 object-cover rounded-xl border-2 border-primary shadow-md" 
              />
              <button 
                onClick={removeImage}
                className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-lg hover:bg-red-600 transition-all transform hover:scale-110"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        <form 
          onSubmit={handleSubmit} 
          className="w-full max-w-4xl mx-auto flex items-center gap-3 bg-gray-50 rounded-2xl border border-gray-200 px-4 py-3 focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all shadow-inner"
        >
          <input 
            type="file" 
            accept="image/*"
            className="hidden"
            ref={fileInputRef}
            onChange={handleImageSelect}
          />
          <button 
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading || isModelLoading || !!modelError}
            className="p-2 text-gray-400 hover:text-primary transition-colors disabled:opacity-50"
            aria-label="Upload image"
          >
            <ImageIcon className="w-6 h-6" />
          </button>

          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading || isModelLoading || !!modelError}
            placeholder={isModelLoading ? "Initializing..." : "Type a medical question..."}
            className="flex-1 bg-transparent border-none outline-none py-1 text-gray-700 disabled:opacity-50"
          />
          <button 
            type="submit" 
            aria-label="Send message"
            disabled={(!input.trim() && !selectedImage) || isLoading || isModelLoading || !!modelError}
            className="p-2.5 rounded-xl text-white bg-primary hover:bg-primary-hover disabled:bg-gray-300 disabled:cursor-not-allowed transition-all transform active:scale-95 shadow-sm"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
        <p className="text-[10px] text-gray-400 text-center mt-3 uppercase tracking-widest font-semibold">
          Powered by Google LiteRT • 100% Private • Medical AI
        </p>
      </footer>
    </div>
  );
}

export default App;
