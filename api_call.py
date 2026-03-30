# api_order_agent.py - Direct API Integration
import requests
import json
from openai import OpenAI

class APIOrderAgent:
    def __init__(self, api_key=None):
        self.base_url = "https://api.akshayakalpa.com/v1"  # Internal API
        self.api_key = api_key
        self.session = requests.Session()
        
    def search_products(self, query):
        """Search products via API"""
        response = self.session.get(
            f"{self.base_url}/products/search",
            params={"q": query}
        )
        return response.json()
    
    def add_to_cart(self, product_id, quantity=1):
        """Add to cart via API"""
        response = self.session.post(
            f"{self.base_url}/cart/add",
            json={"product_id": product_id, "quantity": quantity}
        )
        return response.json()
    
    def get_cart(self):
        """Get current cart"""
        response = self.session.get(f"{self.base_url}/cart")
        return response.json()
    
    def checkout(self, user_details):
        """Place order"""
        response = self.session.post(
            f"{self.base_url}/orders",
            json=user_details
        )
        return response.json()
    
    def voice_order(self, voice_text):
        """Process voice order"""
        # Use LLM to parse voice
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Parse order from voice. Return JSON with products."},
                {"role": "user", "content": voice_text}
            ]
        )
        
        order = json.loads(response.choices[0].message.content)
        
        # Execute order
        for item in order.get("products", []):
            # Search product
            products = self.search_products(item["name"])
            if products:
                product_id = products[0]["id"]
                self.add_to_cart(product_id, item.get("quantity", 1))
        
        # Get cart total
        cart = self.get_cart()
        
        return {
            "order_summary": order,
            "cart_total": cart.get("total"),
            "status": "ready_for_checkout"
        }

# Usage
agent = APIOrderAgent()
result = agent.voice_order("I want 2 A2 milk and 1 paneer")
print(f"Order ready: ₹{result['cart_total']}")
