import json
import random
import os

# Example predefined intents and entities for StrawCore AI booking domains
INTENTS = [
    "hotel_booking",
    "academic_consultation",
    "medical_appointment",
    "payment_query"
]

# A few hardcoded examples to illustrate the data structure.
# In a real scenario, this script would call an API (like OpenAI's GPT-4 or Gemini) 
# to generate thousands of diverse examples.
EXAMPLES = [
    {
        "domain": "hotel_booking",
        "instruction": "The user wants to book a hotel room in Kochi for 2 people.",
        "manglish_input": "Enikku kochi yil oru hotel room book cheyyanam, 2 perundu.",
        "malayalam_input": "എനിക്ക് കൊച്ചിയിൽ ഒരു ഹോട്ടൽ റൂം ബുക്ക് ചെയ്യണം, 2 പേരുണ്ട്.",
        "response": "Sure, I can help you with your hotel booking in Kochi. Ethu divasam aanu booking vendathu? (Which date do you need the booking for?)"
    },
    {
        "domain": "medical_appointment",
        "instruction": "The user is asking to schedule a consultation with a cardiologist.",
        "manglish_input": "Oru cardiologist de appointment edukkanam nale.",
        "malayalam_input": "ഒരു കാർഡിയോളജിസ്റ്റിന്റെ അപ്പോയിന്റ്മെന്റ് എടുക്കണം നാളെ.",
        "response": "Certainly. Nale ethra manikkanu appointment vendathu? (What time do you need the appointment tomorrow?)"
    },
    {
        "domain": "academic_consultation",
        "instruction": "User wants to know the fee for course consultation.",
        "manglish_input": "Course consultation nu ethra aanu fees?",
        "malayalam_input": "കോഴ്സ് കൺസൾട്ടേഷന് എത്രയാണ് ഫീസ്?",
        "response": "The fee for our academic consultation is ₹500. Payment cheyyan ulla link njan thaazhe nalkam. (I'll provide the payment link below.)"
    }
]

def format_for_llama3(system_prompt, user_msg, assistant_msg):
    """
    Formats the conversation using Llama-3 Instruct tags.
    """
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"

def generate_dataset(output_path, num_samples=100):
    system_prompt = "You are StrawCore AI, a highly capable assistant specializing in handling client bookings, consultations, and payments. You understand English, phonetic Manglish, and Malayalam natively."
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Generate mixed manglish and malayalam dataset
        for i in range(num_samples):
            example = random.choice(EXAMPLES)
            
            # Randomly choose between Manglish or pure Malayalam script
            user_input = example["manglish_input"] if random.random() > 0.4 else example["malayalam_input"]
            response = example["response"]
            
            text_block = format_for_llama3(system_prompt, user_input, response)
            
            record = {
                "text": text_block,
                "domain": example["domain"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Generated {num_samples} samples and saved to {output_path}")

if __name__ == "__main__":
    generate_dataset("C:/Users/mhdas/Documents/ManglishLLMmodel/data/manglish_booking_data.jsonl", num_samples=500)
