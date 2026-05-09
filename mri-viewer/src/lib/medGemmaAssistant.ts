import { FilesetResolver, LlmInference } from '@mediapipe/tasks-genai';

export type ModelStatus = 'Loading' | 'Ready' | 'Unavailable' | 'Generating' | 'Error';

type GenerateOptions = {
  prompt: string;
  image?: HTMLImageElement | null;
  onPartial?: (partialText: string) => void;
};

type ModelConfig = {
  displayName: string;
  path: string;
  maxTokens: number;
  maxNumImages: number;
  temperature: number;
  topK: number;
};

const MODEL_CONFIGS = {
  medgemma: {
    displayName: 'MedGemma 1.5 Edge',
    path: '/models/litertlm_medgemma-1.5-4b-it-int4.litertlm',
    maxTokens: 1536,
    maxNumImages: 1,
    temperature: 0.2,
    topK: 40,
  },
  gemma4: {
    displayName: 'Gemma 4 Multimodal',
    path: '/models/gemma-4-E2B-it.litertlm',
    maxTokens: 1536,
    maxNumImages: 1,
    temperature: 0.2,
    topK: 40,
  },
  gemma3Compatibility: {
    displayName: 'Gemma 3 Multimodal',
    path: '/models/gemma-3n-E2B-it-int4-Web.litertlm',
    maxTokens: 1536,
    maxNumImages: 1,
    temperature: 0.2,
    topK: 40,
  },
} satisfies Record<string, ModelConfig>;

function withCacheBust(path: string) {
  return `${path}${path.includes('?') ? '&' : '?'}v=${Date.now()}`;
}

function formatError(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function normalizePrompt(prompt: string) {
  return prompt.trim();
}

async function createLlmInference(
  genai: Awaited<ReturnType<typeof FilesetResolver.forGenAiTasks>>,
  config: ModelConfig
) {
  return LlmInference.createFromOptions(genai, {
    baseOptions: {
      modelAssetPath: withCacheBust(config.path),
    },
    maxTokens: config.maxTokens,
    maxNumImages: config.maxNumImages,
    topK: config.topK,
    temperature: config.temperature,
    randomSeed: Math.floor(Math.random() * 1000),
  });
}

export class MedGemmaAssistantService {
  private genai: Awaited<ReturnType<typeof FilesetResolver.forGenAiTasks>> | null = null;
  private inference: LlmInference | null = null;
  private abortGeneration = false;
  private _modelName = 'MedGemma 1.5 Edge';
  private _modelDetail = 'Preparing local MedGemma runtime...';

  get modelName() {
    return this._modelName;
  }

  get modelDetail() {
    return this._modelDetail;
  }

  async load() {
    this.abortGeneration = false;

    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser. Please use Chrome or Edge with WebGPU enabled.');
    }

    this.genai ??= await FilesetResolver.forGenAiTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm');

    if (this.inference) {
      try {
        this.inference.close();
      } catch (error) {
        console.warn('Failed to close previous MedGemma inference:', error);
      }
      this.inference = null;
    }

    try {
      this._modelDetail = `Loading ${MODEL_CONFIGS.medgemma.displayName} from local model asset...`;
      const primary = await createLlmInference(this.genai, MODEL_CONFIGS.medgemma);

      this.inference = primary;
      this._modelName = MODEL_CONFIGS.medgemma.displayName;
      this._modelDetail = `${MODEL_CONFIGS.medgemma.displayName} is ready. Inference runs locally in this browser.`;
      return;
    } catch (primaryError) {
      const primaryMessage = formatError(primaryError);
      console.warn('MedGemma load failed, trying Gemma 4 fallback:', primaryError);

      try {
        this._modelDetail = `MedGemma load failed (${primaryMessage}). Loading Gemma 4 fallback...`;
        const fallback = await createLlmInference(this.genai, MODEL_CONFIGS.gemma4);

        this.inference = fallback;
        this._modelName = MODEL_CONFIGS.gemma4.displayName;
        this._modelDetail = `${MODEL_CONFIGS.gemma4.displayName} is ready. MedGemma was not accepted by this browser runtime: ${primaryMessage}`;
      } catch (gemma4Error) {
        const gemma4Message = formatError(gemma4Error);
        console.warn('Gemma 4 fallback failed, trying Gemma 3 compatibility fallback:', gemma4Error);

        try {
          this._modelDetail = `Gemma 4 fallback failed (${gemma4Message}). Loading Gemma 3 compatibility fallback...`;
          const compatibilityFallback = await createLlmInference(this.genai, MODEL_CONFIGS.gemma3Compatibility);

          this.inference = compatibilityFallback;
          this._modelName = MODEL_CONFIGS.gemma3Compatibility.displayName;
          this._modelDetail = `${MODEL_CONFIGS.gemma3Compatibility.displayName} is ready. MedGemma failed (${primaryMessage}); Gemma 4 also failed (${gemma4Message}).`;
        } catch (compatibilityError) {
          const compatibilityMessage = formatError(compatibilityError);
          this.inference = null;
          this._modelName = MODEL_CONFIGS.medgemma.displayName;
          this._modelDetail = `MedGemma failed (${primaryMessage}). Gemma 4 fallback also failed (${gemma4Message}). Gemma 3 compatibility fallback failed (${compatibilityMessage}).`;
          throw new Error(this._modelDetail, { cause: compatibilityError });
        }
      }
    }
  }

  async generate({ prompt, image, onPartial }: GenerateOptions) {
    if (!this.inference) {
      throw new Error('MedGemma assistant is not ready.');
    }

    const cleanPrompt = normalizePrompt(prompt);
    if (!cleanPrompt) {
      throw new Error('Prompt text is empty.');
    }

    this.abortGeneration = false;
    let accumulated = '';

    const listener = (partialText: string) => {
      if (this.abortGeneration) return;
      if (accumulated && partialText.startsWith(accumulated)) {
        accumulated = partialText;
      } else {
        accumulated += partialText;
      }
      onPartial?.(accumulated);
    };

    const response = image
      ? await this.inference.generateResponse([{ imageSource: image }, cleanPrompt], listener)
      : await this.inference.generateResponse(cleanPrompt, listener);

    const finalText = typeof response === 'string' ? response.trim() : String(response).trim();
    const result = finalText || accumulated.trim();
    onPartial?.(result);
    return result;
  }

  cancel() {
    this.abortGeneration = true;
  }

  reset() {
    this.cancel();
  }

  close() {
    this.cancel();
    if (!this.inference) return;

    try {
      this.inference.close();
    } catch (error) {
      console.warn('Failed to close MedGemma inference:', error);
    } finally {
      this.inference = null;
    }
  }
}
