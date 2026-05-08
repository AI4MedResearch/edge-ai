import litert_lm
import sys
import os

def test_model(model_path):
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        sys.exit(1)
        
    print(f"--- Loading MedGemma LiteRT-LM: {model_path} ---")
    try:
        # Initialize the Engine
        with litert_lm.Engine(model_path) as engine:
            # Create a Conversation
            with engine.create_conversation() as conversation:
                prompt = "What are the common symptoms of influenza?"
                print(f"User: {prompt}")
                print("Assistant: ", end="", flush=True)
                
                # Perform synchronous inference
                response = conversation.send_message(prompt)
                
                # Extract text from response structure
                # The response structure typically has a 'content' field
                text = response["content"][0]["text"]
                print(text)
                print("\n--- Test Successful ---")
    except Exception as e:
        print(f"\n[Error] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Path relative to medgemma_litert/ directory
    path = "models/litertlm_medgemma-1.5-4b-it-int4.litertlm"
    test_model(path)
