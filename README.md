1. Offline 2D Diagram Extraction (Offline_2d_diagram_extraction.py)
Purpose:
*This script uses a pre-trained vision-language model (smol-vlm-7b) to extract structured information from 2D diagrams provided as image files.

Key Features:
*Loads an image from a local path and resizes it for processing.
*Applies a prompt to extract specific details (e.g., lengths, heights) from the image.
*Uses HuggingFace's AutoProcessor and AutoModelForVisionAndLanguage for vision-language understanding.
*Outputs the extracted data in JSON format.

2. Online 2D Diagram Extraction (online_2d_diagram_extraction.py)
Purpose:
*This script performs a similar function as the offline version but uses an online generative AI model (gemini-2.0-flash-exp) via the google.genai library.
Key Features:
*Integrates with Google’s Generative AI API.
*Processes images and generates responses based on provided prompts.
*Implements safety settings for content generation.
*Handles prompts for extracting features from different diagram views.

3. Metadata Generation
*SQL Metadata (sql_meta_data_generation.py):
Key Features
*NoSQL data model generation
*Integration with LLM Foundation API
*JSON output formatting


4. Offline Table Extraction (table_offline_extraction.py)
Purpose:
*This script identifies and extracts tabular data from images of documents using a YOLO-based object detection model.
Key Features:
*Initializes a YOLO model to detect regions of interest (e.g., tables) in an image.
*Crops detected regions and saves them as separate images.
*Uses a vision-language model to extract data from the cropped table images in both JSON and tabular formats.
*Outputs structured data such as material specifications and dimensions.

5. Online Table Extraction (table_online_extraction.py)
Purpose:
*This script performs table extraction using an online LLM API to interpret detected tables in images.
Key Features:
*Similar workflow to the offline table extraction script but uses a language model API (gpt-4o-mini via LangChain).
*Encodes images in base64 format for API requests.
*Extracts structured data (e.g., plan keys, positions, dimensions) and outputs it as JSON or tables.

