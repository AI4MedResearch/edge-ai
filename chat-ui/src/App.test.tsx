import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import App from './App';

// Mock scrollIntoView as it's not implemented in jsdom
window.HTMLElement.prototype.scrollIntoView = vi.fn();

// Mock navigator.gpu
Object.defineProperty(navigator, 'gpu', {
  value: {},
  configurable: true,
});

// Mock Blob.arrayBuffer and createObjectURL
if (typeof Blob !== 'undefined' && !Blob.prototype.arrayBuffer) {
  Blob.prototype.arrayBuffer = async function() {
    return new ArrayBuffer(this.size);
  };
}

globalThis.URL.createObjectURL = vi.fn().mockReturnValue('mock-url');
globalThis.URL.revokeObjectURL = vi.fn();

// Mock Image
class MockImage {
  onload: () => void = () => {};
  onerror: () => void = () => {};
  src: string = '';
  constructor() {
    setTimeout(() => this.onload(), 0);
  }
}

globalThis.Image = MockImage as unknown as typeof Image;

// Mock @mediapipe/tasks-genai
const mockGenerateResponse = vi.fn();
const mockClose = vi.fn();

vi.mock('@mediapipe/tasks-genai', () => ({
  FilesetResolver: {
    forGenAiTasks: vi.fn().mockResolvedValue({}),
  },
  LlmInference: {
    createFromOptions: vi.fn().mockImplementation(() => Promise.resolve({
    generateResponse: (prompt: string | Array<{ imageSource: HTMLImageElement } | string>, listener?: (partial: string, done: boolean) => void) => {
        if (listener) {
          // Simulate cumulative streaming (Web SDK behavior)
          listener('Part 1 ', false);
          listener('Part 1 Part 2', true);
        }
        return mockGenerateResponse(prompt);
      },
      close: () => mockClose(),
    })),
  },
}));

describe('App Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGenerateResponse.mockReset();
    mockClose.mockReset();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    
    // Mock fetch for model loading with reader support
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-length': '100' }),
      body: {
        getReader: () => {
          let readCount = 0;
          return {
            read: async () => {
              if (readCount === 0) {
                readCount++;
                return { done: false, value: new Uint8Array(100) };
              }
              return { done: true, value: undefined };
            }
          };
        }
      }
    });
  });

  afterEach(() => {
    cleanup();
  });

  it('renders initial loading state and then welcome message', async () => {
    render(<App />);
    expect(screen.getByText('Preparing Local Engine')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('MedGemma 1.5 Edge is Ready')).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('allows user to type and send a text message', async () => {
    mockGenerateResponse.mockResolvedValue('Part 1 Part 2'); // Match the streamed output
    const user = userEvent.setup();
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Type a medical question...')).not.toBeDisabled();
    }, { timeout: 3000 });

    const input = screen.getByPlaceholderText('Type a medical question...');
    const sendButton = screen.getByLabelText('Send message');

    await user.type(input, 'What is diabetes?');
    await user.click(sendButton);

    expect(screen.getByText('What is diabetes?')).toBeInTheDocument();
    
    await waitFor(() => {
      // The mock streams "Part 1 " then "Part 2"
      expect(screen.getByText('Part 1 Part 2')).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('allows user to upload an image and send it', async () => {
    mockGenerateResponse.mockResolvedValue('Part 1 Part 2'); // Match the streamed output
    const user = userEvent.setup();
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByLabelText('Upload image')).not.toBeDisabled();
    }, { timeout: 3000 });

    const file = new File(['hello'], 'test.png', { type: 'image/png' });
    const fileInput = screen.getByLabelText('Upload image').parentElement?.querySelector('input[type="file"]') as HTMLInputElement;
    
    await user.upload(fileInput, file);
    
    expect(screen.getByAltText('Preview')).toBeInTheDocument();

    const sendButton = screen.getByLabelText('Send message');
    await user.click(sendButton);

    expect(screen.getByAltText('Uploaded content')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Part 1 Part 2')).toBeInTheDocument();
    }, { timeout: 3000 });
    
    expect(mockGenerateResponse).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ imageSource: expect.any(Object) }),
        expect.stringContaining('Describe this image.')
      ])
    );
  });

  it('displays error message on generation failure', async () => {
    mockGenerateResponse.mockRejectedValue(new Error('Inference Failed'));
    const user = userEvent.setup();
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Type a medical question...')).not.toBeDisabled();
    }, { timeout: 3000 });

    const input = screen.getByPlaceholderText('Type a medical question...');
    const sendButton = screen.getByLabelText('Send message');
    
    await user.type(input, 'Hi');
    await user.click(sendButton);

    await waitFor(() => {
      expect(screen.getByText(/Error communicating with model: Inference Failed/)).toBeInTheDocument();
    }, { timeout: 3000 });
  });
});
