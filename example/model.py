import json
import requests
import urllib 
from PIL import Image
from io import BytesIO
import base64
import pytesseract
import cv2
import requests
import pymupdf
import openai
from openai import OpenAI
import os
import pandas as pd 

openai_api_key = os.getenv("YOU_OPENAI_API_KEY")


extraction_examples = """
{"is_valid_menu": "yes",
 "menu_complexity": "easy",
 "menu_output": {"categories": [
  {
    "name": "Appetizer",
    "subtitle": "",
    "sort_id": 0, 
    "items":[
      {
        "name": "Salad",
        "description": "your chocie of chicken or pork",
        "is_alcohol": False,
        "is_bike_friendly": True,
        "sort_id": 0,
        "price": 799,
        "extras": [
          {
            "name": "Protein Choice",
            "min_num_options": 1,
          "max_num_options": 1,
          "num_free_options": 0,
          "options": [
                {
                "name": "Chicken",
                "description": "chicken tender",
                "price": 0,
                "sort_id": 0
              },
              {
                "name": "Pork",
                "description": "Pork belly",
                "price": 0,
                "sort_id": 1
              }
            ]
          }
        ]
      },
      {
        "name": "Soup",
        "description": "Warm soup",
        "is_alcohol": False,
        "is_bike_friendly": True,
        "sort_id": 1,
        "price": 499,
        "extras": [
          {
          }
        ]
      }
    ]
  }
]}
}
"""

system_instruction_prompt = """
You are a helpful assistant designed to convert a restaurant menu image into structured JSON data and answer five key questions based on the menu.
"""


menu_extraction_prompt = f"""
** Instructions for Menu Extraction **
You are given an image of a restaurant menu. Your task is to:
Extract all menu-related information from the image and structure it in a JSON format.
  1. Answer five key questions based on the extracted menu.
  2. Your final output should be a JSON object formatted as follows:
{{
  "is_valid_menu": "yes",
  "input_quality": 50,
  "menu_complexity": "easy",
  "menu_output": {{menu_JSON}},
  "confidence": 50
}}

If the image does not contain a valid restaurant menu, menu_JSON should be an empty JSON object: {{}}.


** Five Key Questions to Answer **
1. Is the image a valid restaurant menu?
  -- Answer "yes" or "no".
  -- When there are multiple input images, it's a valid menu as long as one of them has a valid menu. Make your answer based on those valide ones only. If all the inputs are not valid menus, then it's not a valid menu. 
2. On a scale of 0-100, how confident are you in reading the image?
  -- Consider factors such as legibility, completeness, and visibility of the menu.
  -- If the image is not a valid menu, return 0.
3. Is this an "easy" menu?
  -- Answer "easy" or "others".
  -- Answer "easy" if there are no "extras" and "options" in the menu. Refer to Extras & Options sections below on what extras and options mean. 
  -- If there are any extras or options in any items that a consumer can choose from, then it's not an "easy" menu. 
  -- If there is 'build your own' item, it's not an easy menu. 
  -- If there are separate menus for breakfast, lunch, dinner or any other time of the day, it's not an "easy" menu.
  -- If there is "Additionals", "Extras", "Toppings", "Add-ons", etc. in the menu, it's not an "easy" menu.
  -- If there is "choose", "select", "choice of", etc. that suggests any choices, it's not an "easy" menu.
  -- If there is different sizes one can choose from, it's not an "easy" menu.
  -- If not an easy menu, return "others". 
4. Extract all menu items and format them in JSON.
  -- If answer to question 1 is "no" (i.e. is_valid_menu = "no") or answer to question 3 is "others" (i.e. menu_complexity = "others"), no need to build a menu and return empty JSON.
  -- Follow the detailed structure provided below.
5. On a scale of 0-100, how confident are you in the accuracy of the extracted menu?
  If the image is not a valid menu, return 0.


** JSON Format for Menu Extraction **
1. Categories
  - Definition: Categories group similar items (e.g., Appetizers, Entrees, Beverages).
  - Detection: Usually in bold or larger font than menu items. Sometimes prefixed by a heading or section name. Sometimes you might be able to find a description (which will go under subtitle) bottom or top of the menu or each category. 
  - Rules: 
    -- Category description (subtitle) typically can be found next to the category name or bottom of that category section. 
    -- If a category has no no item under it, do not include that category in the menu.
  - Fields:
    {{
      "name": "Appetizers",
      "subtitle": "Light and fresh starters",
      "sort_id": 0,
      "items": [...]
    }}
    name: Category title. If missing, use common sense.
    subtitle: Description (if available).
    sort_id: Index of the category (starting from 0).

2. Items
  - Definition: The actual food/drink items under each category.
  - Detection: Usually listed with a name, description, and price.
  - Rules: 
    -- If multiple items are listed together (e.g. "house salad or cobb salad"; "fries, tots, potatoes"; "salad with chicken or shrimp), split them into separate items. They will have the same price unless specified otherwise.
    -- If multiple drinks are listed together (e.g., "Coke/Sprite"), split them into separate items.
    -- If there is no price for an item, do not include that item in the menu.
    -- If an item is available only on specific days (e.g. Monday only, Monday-Wednesay, Weekend only, etc.), do not include that item in the menu.
    -- If item name spans two or multiple lines. Merge lines if 1) price is assigned on only one line AND 2) the next line does not start a new item but instead completes the previous item. 
  - Fields:
  {{
    "name": "Salad",
    "description": "Your choice of chicken or pork",
    "price": 799,
    "extras": [...]
    "is_alcohol": false,
    "is_bike_friendly": true,
    "sort_id": 0,
  }}
    -- name: Item name.
    -- description: Description of item / additional details (if available).
    -- price: Price in cents (e.g., $4.99 → 499).
    -- extras: Any choices, options, upgrades, add-ons, toppings or substitutions included with an item. See the below Extras & Options section.
    -- is_alcohol: true for alcoholic beverages, otherwise false.
    -- is_bike_friendly: Always true.
    -- sort_id: Index of item within the category.
  - Examples:
    -- If item name is Coke/Sprite and $5, then separate them out. Coke for $5 and Sprite for $5.
    -- If item name spans two lines ("welches cranberry" in one line and "pomagranate" in the next) and only one price is assigned in only one line ($5 in the next line), then this is one item "welches cranberry pomagranate" for $5. 
  
3. Extras & Options
  - Definition of extras: Any choices, options, upgrades, add-ons, toppings or substitutions included with an item.
  - Definition of options: The choices inside of an Extra.
  - Detection: Typically, you will find extras/options information in the item names or description. Sometimes it's available in bottom or top of each category section. Sometimes it's avaialble in category description and all the items under that category share the same extras structure. 
  - Rules on extras: 
    -- Extras require Options. Look at the examples below to get a sense on how this works.
    -- If there are choices (e.g. protein, size, toppings, add-ons) for an item, you need to create an extra.
    -- If there are add-on items you can add to each item (e.g. "make it a meal", etc), they can be added as an extra. 
    -- If a section lists additional items with prices (e.g., "Additionals," "Toppings," "Sides"), interpret it as an extras section.
    -- If a list of price is given for different sizes, 'Size choice' is extra name. 
    -- Each listed extra should be assigned to the relevant category (e.g., pizza toppings under pizza items).
    -- If it's unclear which items they apply to, assume they apply to all items in the category above them.
    -- There could be multiple extras. (e.g. one for Protein Choice, one fro Side Choice)
    -- If no specific options are mentioned, do not create extras.
  - Rules on options:
    -- option name is required but description is not. 
    -- if there is no additional cost or it doesn't say anything about the price, price for an option is typically 0.
    -- "Options" have to be a choice inside of given Extra that a user can choose. If there is no options or only one option to choose, there should not be extras/options.
      
  - Fields:
    {{
      "name": "Protein Choice",
      "min_num_options": 1,
      "max_num_options": 1,
      "num_free_options": 0,
      "options": [
        {{
          "name": "Chicken",
          "description": "Chicken tender",
          "price": 0,
          "sort_id": 0
        }},
        {{
          "name": "Pork",
          "description": "Pork belly",
          "price": 100,
          "sort_id": 1
        }}
      ]
    }}
    -- name: A generic label (e.g., Size Choice, Protein Choice).
    -- min_num_options: Minimum required selections.
    -- max_num_options: Maximum allowed selections.
    -- num_free_options: How many are free before an extra charge.
    -- options: List of choices under the extra.  
      --- Under Options, for each option, there are 
      --- name: Choice name.
      --- description: Additional details (if available).
      --- price: additional price to add to the price of the item.
      --- sort_id: Index of option within the extra.
  - Examples:
    -- If description says 'choice of pork or chicken', then your item has 'extras' and the name can be 'Protein Choice'. Options will be 'pork' and 'chicken' 
    -- If an item offers different sizes (for example, small/large or 8inch/12inch pizza, etc.), "extra" can be 'Size Choice' and option will be different sizes.
    -- If item desctiption says 'your choice of meat', but no choice of meat is specified, then do not create extras for this item
    -- If it says 'make it a combo for $3.99', then extra name can be "Preparation Option" and option name can be "Make it a combo" and price is 399.
    -- If an item comes in different sizes, flavors, or variations (e.g., 'Buffalo Wings' with different piece options: 5, 10, 20).
    -- If an item offers different sizes (for example, small/larger or 8inch/12inch pizza, etc.), "extra" can be 'Size Choice' and option will be different sizes.
    -- If an item has additional toppings to add (for example, cheese, tomato, onion, etc.), "extra" can be 'Toppings Choice' and option will be different toppings. If there is additional cost, add that in price. 
    -- Typically a given menu does not have names for extras. You come up with a name like 'Size Choice', 'Protein Choice', 'Side Choice', 'Add Ons', etc. Use a common sense.
    -- If the description includes 'your choice of pork or chicken', then min_num_option=1, max_num_option=1, num_free_options=0. 
    -- If it says 'choose 2', then min_num_option=2, max_num_option=2, num_free_options=0.
    -- If it says 'choose up to 2 choices', then min_num_option=0, max_num_option=2, num_free_options=0.
    

** Processing Rules ** 
- General Rules
  -- When there are multiple input images given, it's a valid menu as long as one of them has valid menu. Make your answer based on those valide ones only. If all the inputs are not valid menu, then it's not a valid menu. 
  -- Focus on text and numbers of available items. If there is food image, do not try to anlayze it. 
  

- General Rules on names and descriptions
  -- Keep the name and description as in the given menu. Keep all special characters or numbers (e.g. keep the full name for #4 Soup, 12. hamburger, dumpling (1), w/ beef) as in the given menu. 
  -- Preserve non-English words as they appear. If category/item name is written in English and another langauge, preserve both langugaes as they appear in the menu.
  -- If there is '*' next to the item or category, look for what this * means and add it to description. 
  -- Use common sense on category names or extra names when they are not specified in the menu.
  
- Price Handling
  -- Convert all prices to cents (e.g., $4.99 → 499). Usually price in the image is in dollars. (e.g. 11 -> 1100)
  -- If a price is per pound, append "(per lb)" to the name and show only numbers in price. 
  -- If price is not presented next to an item, look for a price around category name and apply it at the item level if it's available. 
  -- If item's price still isn't clear, don't include that item.
  -- Drop any items with no price. Do not arbitrarily assign a price to an item when price is not available.

- Special Symbols & Formatting
  -- If an item has * next to it, find what it means and add it to the description.
  -- If there is a pepper image sign next to an item, add 'Spicy' to the description. 
  -- If a drink is named "Coca", convert it to "Coca-Cola".
  

- Handling Alcoholic Beverages
  -- Do not include alcoholic beverages in the extracted menu.

** Example Output **
{{
  "is_valid_menu": "yes",
  "input_quality": 80,
  "menu_complexity": "others",
  "menu_output": {extraction_examples},
  "confidence": 90
}}

"""



class ChatCompletionAgent:
    def __init__(self, model_name, openai_api_key, temperature, response_format):
        self.model_name = model_name
        self.temperature = temperature
        self.response_format = response_format
        self.client = OpenAI(
            api_key = openai_api_key, 
            organization = "org-HvVpqVsX21frElw6ih05m7aS"
        )

    def get_response(self, messages):
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": self.response_format
        }    
        completion = self.client.chat.completions.create(**params)
        response = completion.choices[0].message.content
            
        return response


class menuBuilder:
    def __init__(self, system_instruction_prompt, menu_extraction_prompt, extraction_examples, chat_agent):
        self.extraction_examples = extraction_examples
        self.system_instruction_prompt = system_instruction_prompt
        self.menu_extraction_prompt = menu_extraction_prompt  
        self.chat_agent = chat_agent

    
    def fetch_url_content(self, url):
        """Fetches the content of a URL and determines its file type."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
            response = requests.get(url, headers=headers, allow_redirects=True)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch URL: {url}, Status Code: {response.status_code}")

            content_type = response.headers.get("Content-Type", "").lower()
            
            if "application/pdf" in content_type:
                return "pdf", response.content
            elif "image" in content_type:
                return "image", response.content

        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return "unknown", None

    def pdf_to_base64_images(self, pdf_content):
        """Converts a PDF (bytes) into a list of base64-encoded images."""
        base64_images = []
        pdf_stream = BytesIO(pdf_content)
        pdf_document = pymupdf.open(stream=pdf_stream, filetype="pdf")  # Open PDF in memory

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # High resolution (2x scale)
            img = BytesIO(pix.tobytes("png"))  # Convert to PNG
            base64_images.append({"image": base64.b64encode(img.getvalue()).decode("utf-8")})

        return base64_images


    def correct_image_rotation(self, image_bytes):
        """Detects and corrects 90°, 180°, or 270° rotated images using pytesseract."""
        # Convert bytes to PIL Image
        img_pil = Image.open(BytesIO(image_bytes))
        try:
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

            # Use Tesseract to detect orientation
            osd = pytesseract.image_to_osd(img_cv, output_type=pytesseract.Output.DICT)
            angle = osd.get("rotate", 0)  # Get the detected rotation angle
            orientation_conf = osd.get("orientation_conf", 0) # Get the confidence level

            # Rotate the image by the detected angle (negative to correct it)
            if (orientation_conf > 5) and (angle != 0):  # Only rotate if necessary
                img_pil = img_pil.rotate(-angle, expand=True)  
            return img_pil
        except:
            print('tesseract error')
            return img_pil

    def image_to_base64(self, image):
        """Converts a PIL image to base64."""
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")
      

    def generate_gpt_messages(self, system_instruction_prompt, menu_extraction_prompt, extraction_examples, urls):
        
        base64_images = []

        for url in urls:
            file_type, content = self.fetch_url_content(url)

            if file_type == "image":
                rotated_image = self.correct_image_rotation(content)  # Correct rotation
                base64_images.append({"image": self.image_to_base64(rotated_image)})
            elif file_type == "pdf":
                base64_images.extend(self.pdf_to_base64_images(content))  # Convert PDF to images
            else:
                print(f"Skipping unsupported file type for URL: {url}")

        messages = [
            {"role": "system", "content": system_instruction_prompt},
            {
                "role": "user",
                "content": [menu_extraction_prompt] + base64_images
            }
        ]
        
        return messages

    def menu_builder(self, urls):
        messages = self.generate_gpt_messages(self.system_instruction_prompt, self.menu_extraction_prompt, self.extraction_examples, urls)  
        response = self.chat_agent.get_response(messages)

        return response
      


# remove any items with zero or null price
def remove_items_with_zero_or_null_price(menu_json):
    for category in menu_json.get("categories", []):
        category["items"] = [item for item in category["items"] if item.get("price") not in (0, None)]
    return menu_json
  
# remove a category if it has no items under it
def remove_empty_categories(menu_json):
    if "categories" in menu_json:
        menu_json["categories"] = [category for category in menu_json.get("categories", []) if category.get("items")]
    return menu_json      


# Function to convert JSON to the specified flat format
def json_to_flat_format(json_data):
    
    # return empty dataframe if input JSON is empty  
    column_names = ['type', 'name', 'description', 'price', 'num_min_options', 'num_max_options', 'num_free_options']
    df = pd.DataFrame(columns=column_names)
    if len(json_data) == 0 or len(json_data['categories']) == 0:
        return df
    
    # Traverse the JSON hierarchy if input JSON is not empty
    rows = []

    for category in json_data.get('categories', []):
        category_name = category['name']
        # Add category row
        rows.append({
            'type': 'Category',
            'name': category_name,
            'description': '',
            'price': '',
            'num_min_options': '',
            'num_max_options': '',
            'num_free_options': ''
        })

        for item in category.get('items', []):
            item_name = item['name']
            item_description = item.get('description', '')
            item_price = item.get('price', 0)
            # Add item row
            rows.append({
                'type': 'Item',
                'name': item_name,
                'description': item_description,
                'price': item_price,
                'num_min_options': '',
                'num_max_options': '',
                'num_free_options': ''
            })

            for extra in item.get('extras', []):
                extra_name = extra['name']
                min_options = extra.get('min_num_options', 0)
                max_options = extra.get('max_num_options', 0)
                free_options = extra.get('num_free_options', 0)
                # Add extra row
                rows.append({
                    'type': 'Extra',
                    'name': extra_name,
                    'description': '',
                    'price': '',
                    'num_min_options': min_options,
                    'num_max_options': max_options,
                    'num_free_options': free_options
                })

                for option in extra.get('options', []):
                    option_name = option['name']
                    option_description = option.get('description', '')
                    option_price = option.get('price', 0)
                    # Add option row
                    rows.append({
                        'type': 'Option',
                        'name': option_name,
                        'description': option_description,
                        'price': option_price,
                        'num_min_options': '',
                        'num_max_options': '',
                        'num_free_options': ''
                    })

    # Create a DataFrame
    return pd.DataFrame(rows)



# Open AI setup 
temperature = 0.0
response_format = {"type": "json_object"}
model_name = 'gpt-4o'

# initialize the model
chat_agent = ChatCompletionAgent(model_name, openai_api_key, temperature, response_format)
menu_builder_model = menuBuilder(system_instruction_prompt, menu_extraction_prompt, extraction_examples, chat_agent)

