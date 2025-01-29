import os
from dotenv import load_dotenv
from analyzer import OutfitAnalyzerChain

def main():
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")
    
    try:
        analyzer = OutfitAnalyzerChain(api_key)
        analyzer.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()